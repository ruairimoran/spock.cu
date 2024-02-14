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
	DeviceArray dev_data = DeviceArray<float, 3>::alloc();

	DeviceArray cloned_data = dev_data.clone();
	EXPECT_EQ(dev_data.size(), cloned_data.size());
	
	std::array<float, 3> downloaded_host_data;
	cloned_data.download(downloaded_host_data);

	for (int i = 0; i < 3; i++) {
		EXPECT_EQ(downloaded_host_data[i], host_data[i]);
	}
}
