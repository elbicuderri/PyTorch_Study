#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <chrono>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

template <typename T>
std::vector<T> load_data(const std::string &file_name)
{
	std::ifstream input(file_name, std::ios::in | std::ios::binary);
	if (!(input.is_open()))
	{
		std::cout << "Cannot open the " << file_name << std::endl;
		std::exit(-1);
	}

	std::vector<T> data;
	input.seekg(0, std::ios::end);
	auto size = input.tellg();
	input.seekg(0, std::ios::beg);

	for (size_t i = 0; i < static_cast<size_t>(size) / sizeof(T); ++i)
	{
		T value;
		input.read((char *)&value, sizeof(T));
		data.push_back(value);
	}

	return data;
}

template <typename T>
T Area(const std::vector<T> &box)
{
	T x1 = box[0];
	T y1 = box[1];
	T x2 = box[2];
	T y2 = box[3];

	assert(x2 >= x1 && y2 >= y1);

	T width = x2 - x1;
	T height = y2 - y1;

	T area = width * height;

	return area;
}

template <typename T>
T IOU(const std::vector<T> &box1, const std::vector<T> &box2)
{
	T box1_x1 = box1[0];
	T box1_y1 = box1[1];
	T box1_x2 = box1[2];
	T box1_y2 = box1[3];

	T box2_x1 = box2[0];
	T box2_y1 = box2[1];
	T box2_x2 = box2[2];
	T box2_y2 = box2[3];

	T left_top_x = std::max(box1_x1, box2_x1);
	T left_top_y = std::max(box1_y1, box2_y1);

	T right_bottom_x = std::min(box1_x2, box2_x2);
	T right_bottom_y = std::min(box1_y2, box2_y2);

	T inter_width = std::max(right_bottom_x - left_top_x, static_cast<T>(0));
	T inter_height = std::max(right_bottom_y - left_top_y, static_cast<T>(0));

	T inter_area = inter_width * inter_height;

	T box1_area = Area(box1);
	T box2_area = Area(box2);

	T iou = inter_area / (box1_area + box2_area - inter_area);

	return iou;
}

template <typename T>
std::vector<int> arg_sort(const std::vector<T> &given_vector)
{
	std::vector<int> sorted_args(given_vector.size());
	std::iota(sorted_args.begin(), sorted_args.end(), 0);
	std::sort(sorted_args.begin(), sorted_args.end(),
						[&given_vector](int left, int right) -> bool {
							return given_vector[left] > given_vector[right];
						});

	return sorted_args;
}

template <typename T>
std::vector<int> nms_cpu(const std::vector<std::vector<T>> &boxes, const std::vector<T> &scores, T threshold)
{

	std::vector<int> keep_indices = arg_sort<T>(scores);
	std::vector<int> meta_index = keep_indices;

	while (meta_index.size() > 0)
	{
		int idx_self = meta_index[0];
		// meta_index[1:]
		std::vector<int> idx_others(std::move(std::vector<int>(meta_index.begin() + 1, meta_index.end())));

		for (const auto& idx_other: idx_others)
		{
			T iou = IOU<T>(boxes[idx_self], boxes[idx_other]);
			if (iou > threshold)
			{
				meta_index.erase(std::remove(meta_index.begin(), meta_index.end(), idx_other), meta_index.end());
				keep_indices.erase(std::remove(keep_indices.begin(), keep_indices.end(), idx_other), keep_indices.end());
			}
		}		
		meta_index.erase(meta_index.begin());
	}

	return keep_indices;
}

template <typename T>
std::vector<int> nms_cpu_v2(const std::vector<T> &boxes, const std::vector<T> &scores, T threshold)
{
	std::vector<int> keep_indices = arg_sort<T>(scores);
	std::vector<int> meta_index = keep_indices;

	while (meta_index.size() > 0)
	{
		int idx_self = meta_index[0];
		// meta_index[1:]
		std::vector<int> idx_others(std::move(std::vector<int>(meta_index.begin() + 1, meta_index.end())));
		for (const auto& idx_other: idx_others)
		{
			std::vector<T> box1(std::move(std::vector<T>(boxes.begin() + 4 * idx_self, boxes.begin() + 4 * idx_self + 4)));
			std::vector<T> box2(std::move(std::vector<T>(boxes.begin() + 4 * idx_other, boxes.begin() + 4 * idx_other + 4)));
			T iou = IOU<T>(box1, box2);
			if (iou > threshold)
			{
				meta_index.erase(std::remove(meta_index.begin(), meta_index.end(), idx_other), meta_index.end());
				keep_indices.erase(std::remove(keep_indices.begin(), keep_indices.end(), idx_other), keep_indices.end());
			}
		}		
		meta_index.erase(meta_index.begin());
	}

	return keep_indices;
}

int main()
{
	// std::vector<float> data = load_data<float>("nms_test_bboxes.bin");
	// std::vector<float> scores = load_data<float>("nms_test_scores.bin");
	// std::vector<int> result = load_data<int>("nms_test_out.bin");

	std::vector<float> data = load_data<float>("nms_test_bboxes_small.bin");
	std::vector<float> scores = load_data<float>("nms_test_scores_small.bin");
	std::vector<int> result = load_data<int>("nms_test_out_small.bin");

	std::cout << "Size of data: " << data.size() << std::endl;
	std::cout << "Size of scores: " << scores.size() << std::endl;
	std::cout << "Size of result: " << result.size() << std::endl;

	std::cout << "Data loaded!!\n\n";

	//==================================================================================
	std::vector<std::vector<float>> boxes;
	auto start_data_setting = std::chrono::steady_clock::now();
	for (int i = 0; i < scores.size(); ++i)
	{
		std::vector<float> tmp = std::vector<float>(data.begin() + 4*i, data.begin() + 4*i + 4);
		boxes.push_back(tmp);
	}
	auto end_data_setting = std::chrono::steady_clock::now();
	std::cout << "Elapsed time in microseconds : "
	<< std::chrono::duration_cast<std::chrono::microseconds>(end_data_setting - start_data_setting).count()
	<< " micro-sec\n";
	std::cout << "Data ready!!\n\n";
	//==================================================================================
	auto start = std::chrono::steady_clock::now();
	std::vector<int> keep = nms_cpu<float>(boxes, scores, 0.5f);
	auto end = std::chrono::steady_clock::now();

	std::cout << "Elapsed time in microseconds : "
		<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
		<< " micro-sec\n";
	//==================================================================================
	auto start_v2 = std::chrono::steady_clock::now();
	std::vector<int> keep_v2 = nms_cpu_v2<float>(data, scores, 0.5f);
	auto end_v2 = std::chrono::steady_clock::now();

	std::cout << "Elapsed time in microseconds : "
		<< std::chrono::duration_cast<std::chrono::microseconds>(end_v2 - start_v2).count()
		<< " micro-sec\n";
	//==================================================================================
	std::cout << "Size of keep: " << keep.size() << std::endl;
	// for (auto i = 0; i < result.size(); ++i) {
	// 	if (keep[i] != static_cast<int>(result[i])) {
	// 		std::cout << "wrong!!" << std::endl;
	// 		std::cout << "origin result " << result[i] << " != " << "my result: " << keep[i] << std::endl;
	// 		return 0;
	// 	}
	// }
	// std::cout << "Success!!" << std::endl;

	if (keep == result) {
		std::cout << "Correct!!" << std::endl;
	} else {
		std::cout << "Wrong!!" << std::endl;
	}

	// std::cout << "Size of keep: " << keep_v2.size() << std::endl;
	// for (auto i = 0; i < result.size(); ++i) {
	// 	if (keep_v2[i] != static_cast<int>(result[i])) {
	// 		std::cout << "wrong!!" << std::endl;
	// 		std::cout << "origin result " << result[i] << " != " << "my result: " << keep_v2[i] << std::endl;
	// 		return 0;
	// 	}
	// }
	// std::cout << "Version 2 Success!!" << std::endl;

	if (keep_v2 == result) {
		std::cout << "Version 2 Correct!!" << std::endl;
	} else {
		std::cout << "Wrong!!" << std::endl;
	}	

	

	return 0;
}