/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// CPU-sim TASSIGN tests:
//   - TAssignAliasTest:  an integer argument is a byte offset, a host pointer into
//     a simulated region is an alias to that storage.
//   - TAssignWindowTest: a windowed sub-view (full block shape, smaller valid
//     shape, non-zero byte offset) resolves against its valid-region footprint.

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
// A2/A3 L0C is 128 KB. A [128,256] int32 accumulator is exactly 128 KB, so it
// fills the whole region and leaves zero head-room for a non-zero strip offset.
constexpr std::size_t kL0CBytes = 128 * 1024;
constexpr int kRows = 128;
constexpr int kCols = 256;
constexpr int kStripRows = 16;

// Full accumulator: 128 rows valid (dynamic valid shape, exercises the Numel
// fallback path) placed at offset 0 -- the exact-fit boundary case.
using AccFull = Tile<TileType::Acc, int32_t, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
// 16-row strip view: static valid shape 16x256, the windowed sub-view PTOAS pushes.
using AccStrip =
    Tile<TileType::Acc, int32_t, kRows, kCols, BLayout::ColMajor, kStripRows, kCols, SLayout::RowMajor, 1024>;
} // namespace

class TAssignAliasTest : public testing::Test {
protected:
    void SetUp() override
    {
        NPU_MEMORY_INIT(NPUArch::A2A3);
        NPU_MEMORY_CLEAR();
    }
};

TEST_F(TAssignAliasTest, exact_alias_reuses_same_backing_buffer)
{
    using BaseTile = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64>;
    using AliasTile = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64>;

    BaseTile base;
    AliasTile alias;

    TASSIGN(base, 0);
    TASSIGN(alias, reinterpret_cast<std::uintptr_t>(base.data()));

    ASSERT_EQ(base.data(), alias.data());

    base.SetValue(5, 3.5f);
    EXPECT_FLOAT_EQ(alias.GetValue(5), 3.5f);

    alias.SetValue(17, -2.25f);
    EXPECT_FLOAT_EQ(base.GetValue(17), -2.25f);
}

TEST_F(TAssignAliasTest, interior_alias_treats_pointer_as_pointer_not_offset)
{
    using BaseTile = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64>;
    using TailTile = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64>;

    constexpr std::size_t kRowOffset = 4;
    constexpr std::size_t kColCount = 64;
    constexpr std::size_t kElementOffset = kRowOffset * kColCount;

    BaseTile base;
    TailTile tail;

    TASSIGN(base, 0);
    TASSIGN(tail, reinterpret_cast<std::uintptr_t>(base.data() + kElementOffset));

    ASSERT_EQ(tail.data(), base.data() + kElementOffset);

    tail.SetValue(3, 9.0f);
    EXPECT_FLOAT_EQ(base.GetValue(kElementOffset + 3), 9.0f);

    base.SetValue(kElementOffset + 10, -7.0f);
    EXPECT_FLOAT_EQ(tail.GetValue(10), -7.0f);
}

class TAssignWindowTest : public testing::Test {
protected:
    void SetUp() override
    {
        NPU_MEMORY_INIT(NPUArch::A2A3);
        NPU_MEMORY_CLEAR();
    }
};

// The full 128x256 accumulator exactly fills L0C; assigning it at offset 0 must
// still be accepted (valid-region footprint equals the full block Numel here).
TEST_F(TAssignWindowTest, full_accumulator_exactly_fills_l0c)
{
    auto* base = NPUMemoryModel::Instance().GetL0CBase();
    ASSERT_NE(base, nullptr);

    AccFull full(kRows, kCols);
    TASSIGN(full, static_cast<uint64_t>(0));
    EXPECT_EQ(reinterpret_cast<char*>(full.data()), base);
}

// Each 16-row strip is TASSIGN'd at byte offset rowStart * kStripRows * sizeof,
// matching the PTOAS-generated offsets (0, 1024, 2048, ... for int32). A strip
// addresses only its 16 valid rows, so every offset stays in-bounds and resolves
// to a pointer inside L0C.
TEST_F(TAssignWindowTest, windowed_strip_offsets_stay_in_bounds)
{
    auto* base = NPUMemoryModel::Instance().GetL0CBase();
    ASSERT_NE(base, nullptr);

    for (int rowStart = 0; rowStart < kRows; rowStart += kStripRows) {
        const std::size_t byteOffset =
            static_cast<std::size_t>(rowStart) * static_cast<std::size_t>(kStripRows) * sizeof(int32_t);

        AccStrip strip;
        TASSIGN(strip, static_cast<uint64_t>(byteOffset));

        auto* ptr = reinterpret_cast<char*>(strip.data());
        EXPECT_EQ(ptr, base + byteOffset) << "strip offset " << byteOffset;
        // The strip's whole valid-region footprint must lie within the L0C buffer.
        const std::size_t footprintBytes =
            (static_cast<std::size_t>(GetTileOffset<AccStrip>(kStripRows - 1, kCols - 1)) + 1) * sizeof(int32_t);
        EXPECT_LE(byteOffset + footprintBytes, kL0CBytes) << "strip offset " << byteOffset;
    }
}

// Writing through a windowed strip and reading back via the raw L0C base confirms
// the resolved pointer addresses real, in-bounds backing storage (no overflow).
TEST_F(TAssignWindowTest, windowed_strip_roundtrips_through_backing_storage)
{
    auto* base = reinterpret_cast<int32_t*>(NPUMemoryModel::Instance().GetL0CBase());
    ASSERT_NE(base, nullptr);

    constexpr std::size_t byteOffset =
        static_cast<std::size_t>(kStripRows) * static_cast<std::size_t>(kStripRows) * sizeof(int32_t); // 1024
    AccStrip strip;
    TASSIGN(strip, static_cast<uint64_t>(byteOffset));

    const std::size_t elemOffset = byteOffset / sizeof(int32_t);
    strip.data()[0] = 0x5a5a5a5a;
    EXPECT_EQ(base[elemOffset], 0x5a5a5a5a);
}

TEST_F(TAssignWindowTest, every_strip_reads_the_parent_layout)
{
    AccFull full(kRows, kCols);
    TASSIGN(full, 0);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            full.SetElement(row, col, row * kCols + col);
        }
    }
    for (int rowStart = 0; rowStart < kRows; rowStart += kStripRows) {
        const std::size_t offset = GetTileOffset<AccFull>(rowStart, 0);
        AccStrip strip;
        AccStrip alias;
        TASSIGN(strip, offset * sizeof(int32_t));
        TASSIGN(alias, reinterpret_cast<std::uintptr_t>(full.data() + offset));
        for (int row = 0; row < kStripRows; ++row) {
            for (int col = 0; col < kCols; ++col) {
                EXPECT_EQ(strip.GetElement(row, col), full.GetElement(rowStart + row, col));
                EXPECT_EQ(alias.GetElement(row, col), full.GetElement(rowStart + row, col));
            }
        }
    }
}

TEST_F(TAssignWindowTest, rejects_a_view_that_reaches_past_l0c)
{
    // The last legal strip ends exactly at the L0C boundary. Advancing it by
    // one element must still fail, including when the address is a host alias.
    const std::size_t offset = GetTileOffset<AccFull>(kRows - kStripRows, 0) * sizeof(int32_t);
    AccStrip strip;
    EXPECT_DEATH(TASSIGN(strip, offset + sizeof(int32_t)), "exceeds memory region capacity");
    auto* base = NPUMemoryModel::Instance().GetL0CBase();
    EXPECT_DEATH(
        TASSIGN(strip, reinterpret_cast<std::uintptr_t>(base + offset + sizeof(int32_t))),
        "exceeds memory region capacity");
}

TEST_F(TAssignWindowTest, dynamic_and_padded_tiles_still_require_the_full_box)
{
    AccFull dynamic(kStripRows, kCols);
    EXPECT_DEATH(TASSIGN(dynamic, sizeof(int32_t)), "exceeds memory region capacity");

    using Padded = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Zero>;
    Padded padded;
    // A padded load can touch all 16 rows even with one valid row.
    EXPECT_DEATH(TASSIGN(padded, 192 * 1024 - 64 * sizeof(float)), "exceeds memory region capacity");
}

TEST_F(TAssignWindowTest, host_storage_views_use_the_same_span_check)
{
    auto storage = std::make_shared<std::vector<int32_t>>(kRows * kCols);
    NPUMemoryModel::Instance().RegisterHostStorage(storage);
    const std::size_t offset = GetTileOffset<AccFull>(kRows - kStripRows, 0);
    AccStrip strip;
    TASSIGN(strip, reinterpret_cast<std::uintptr_t>(storage->data() + offset));
    strip.SetElement(kStripRows - 1, kCols - 1, 42);
    EXPECT_EQ(storage->back(), 42);
    EXPECT_DEATH(
        TASSIGN(strip, reinterpret_cast<std::uintptr_t>(storage->data() + offset + 1)),
        "exceeds host storage capacity");
}

TEST_F(TAssignWindowTest, packed_view_span_counts_bytes_instead_of_logical_elements)
{
    using Packed = Tile<TileType::Vec, float4_e2m1x2_t, 4, 64, BLayout::RowMajor, 1, 64>;
    static_assert(Packed::kPackedElementsPerByte == 2);
    const std::size_t offset = 192 * 1024 - 32;
    Packed view;
    TASSIGN(view, offset);
    EXPECT_EQ(reinterpret_cast<char*>(view.data()), NPUMemoryModel::Instance().GetUBBase() + offset);
    EXPECT_DEATH(TASSIGN(view, offset + 1), "exceeds memory region capacity");
}
