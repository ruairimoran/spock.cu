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
