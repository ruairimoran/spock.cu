#include <array>

#include <gtest/gtest.h>

#include "../src/device_array.cuh"

TEST(DeviceArray, Allocate) {
	DeviceArray array_fixed = DeviceArray<float, 123>::alloc();
	EXPECT_EQ(array_fixed.size(), 123);
	EXPECT_EQ(array_fixed.size_bytes(), 123 * sizeof(float));

	DeviceArray array_dyn = DeviceArray<float>::alloc(456);
	EXPECT_EQ(array_dyn.size(), 456);
	EXPECT_EQ(array_dyn.size_bytes(), 456 * sizeof(float));
}

TEST(DeviceArray, Transfer) {
	const std::array<float, 3> host_data = { 123.0f, 456.0f, 789.0f };
	DeviceArray dev_data = DeviceArray<float, 3>::alloc();

	dev_data.upload(host_data);

	std::vector<float> download_dest;
	download_dest.resize(dev_data.size());

	ASSERT_EQ(host_data.size(), download_dest.size());

	dev_data.download(download_dest);

	for (size_t i = 0; i < host_data.size(); i++) {
		EXPECT_EQ(host_data[i], download_dest[i]);
	}
}

TEST(DeviceArray, Empty) {
	DeviceArray array_1 = DeviceArray<float>();
	EXPECT_TRUE(array_1.empty());
	EXPECT_EQ(array_1.data(), nullptr);
	EXPECT_EQ(array_1.size(), 0);
	EXPECT_EQ(array_1.size_bytes(), 0);

	DeviceArray array_2 = DeviceArray<float>::alloc(0);
	EXPECT_TRUE(array_2.empty());
	EXPECT_EQ(array_2.data(), nullptr);
	EXPECT_EQ(array_2.size(), 0);
	EXPECT_EQ(array_2.size_bytes(), 0);
}

TEST(DeviceArray, Getters) {
	DeviceArray array = DeviceArray<float, 3>::alloc();
	std::span<float, 3> array_span = array.get();
	EXPECT_EQ(array_span.data(), array.data());
	EXPECT_EQ(array_span.size(), array.size());
}

TEST(DeviceArray, Clone) {
	const std::array<float, 3> host_data = { 123.0f, 456.0f, 789.0f };
	DeviceArray dev_data = DeviceArray<float>::alloc(3);
	dev_data.upload(host_data);

	DeviceArray cloned_data = dev_data.clone();
	EXPECT_EQ(dev_data.size(), cloned_data.size());
	
	std::array<float, 3> downloaded_host_data = {};
	cloned_data.download(downloaded_host_data);

	for (int i = 0; i < 3; i++) {
		EXPECT_EQ(downloaded_host_data[i], host_data[i]);
	}
}

TEST(DeviceSlice, Create) {
	const std::array<float, 10> host_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
	DeviceArray dev_data = DeviceArray<float>::alloc(host_data.size());
	dev_data.upload(host_data);

	DeviceSlice<float> implicit_slice = dev_data;
	EXPECT_EQ(implicit_slice.data(), dev_data.data());
	EXPECT_EQ(implicit_slice.size(), dev_data.size());
	EXPECT_EQ(implicit_slice.size_bytes(), dev_data.size_bytes());

	DeviceSlice<float> explicit_full_slice = dev_data.slice(0);
	EXPECT_EQ(explicit_full_slice.data(), dev_data.data());
	EXPECT_EQ(explicit_full_slice.size(), dev_data.size());
	EXPECT_EQ(explicit_full_slice.size_bytes(), dev_data.size_bytes());

	DeviceSlice<float> part_slice_upper = dev_data.slice(5);
	EXPECT_EQ(part_slice_upper.data(), dev_data.data() + 5);
	EXPECT_EQ(part_slice_upper.size(), 5);
	EXPECT_EQ(part_slice_upper.size_bytes(), 5 * sizeof(float));

	DeviceSlice<float> part_slice_middle = dev_data.slice(2, 4);
	EXPECT_EQ(part_slice_middle.data(), dev_data.data() + 2);
	EXPECT_EQ(part_slice_middle.size(), 4);
	EXPECT_EQ(part_slice_middle.size_bytes(), 4 * sizeof(float));
}

TEST(DeviceSlice, Transfer) {
	const std::array<float, 10> host_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
	DeviceArray dev_data = DeviceArray<float>::alloc(host_data.size());
	dev_data.upload(host_data);

	DeviceSlice<float> slice = dev_data.slice(2, 4);
	std::array<float, 4> downloaded_slice = {};
	slice.download(downloaded_slice);

	for (int i = 0; i < 4; i++) {
		EXPECT_EQ(host_data[i + 2], downloaded_slice[i]);
	}

	std::array<float, 4> replacement_data = { 6.0, 5.0f, 4.0f, 3.0f };
	slice.upload(replacement_data);

	std::array<float, 10> downloaded_host_data = {};
	dev_data.download(downloaded_host_data);

	const std::array<float, 10> test_host_data = { 1.0f, 2.0f, 6.0f, 5.0f, 4.0f, 3.0f, 7.0f, 8.0f, 9.0f, 10.0f };

	for (int i = 0; i < 10; i++) {
		EXPECT_EQ(test_host_data[i], downloaded_host_data[i]);
	}
}

TEST(DeviceSlice, Subslice) {
	const std::array<float, 10> host_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
	DeviceArray dev_data = DeviceArray<float>::alloc(host_data.size());
	dev_data.upload(host_data);

	DeviceSlice<float> slice = dev_data.slice(3, 5);

	DeviceSlice<float> subslice = slice.slice(2, 3);
	std::array<float, 3> downloaded_subslice = {};
	subslice.download(downloaded_subslice);

	for (int i = 0; i < 3; i++) {
		EXPECT_EQ(host_data[i + 5], downloaded_subslice[i]);
	}
}

TEST(DeviceSlice, Clone) {
	const std::array<float, 10> host_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
	DeviceArray dev_data = DeviceArray<float>::alloc(host_data.size());
	dev_data.upload(host_data);

	DeviceSlice<float> slice = dev_data.slice(3, 5);
	DeviceArray<float> cloned_slice = slice.clone();

	EXPECT_EQ(slice.size(), cloned_slice.size());
	
	std::array<float, 5> downloaded_slice_data = {};
	cloned_slice.download(downloaded_slice_data);

	for (int i = 0; i < 5; i++) {
		EXPECT_EQ(downloaded_slice_data[i], host_data[i + 3]);
	}

	const std::array<float, 10> new_host_data = { 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
	dev_data.upload(new_host_data);

	cloned_slice.download(downloaded_slice_data);

	for (int i = 0; i < 5; i++) {
		EXPECT_EQ(downloaded_slice_data[i], host_data[i + 3]);
	}
}
