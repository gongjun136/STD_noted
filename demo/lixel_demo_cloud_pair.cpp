/**
 * @file lixel_demo.cpp
 * @author your name (you@domain.com)
 * @brief 用于加载对比两个pcd点云细致了解
 * @version 0.1
 * @date 2023-09-21
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "include/STDesc2.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

namespace fs = boost::filesystem;

std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr>
load_pcds_from_directory(const std::string &directory_path, const std::string &source_name, const std::string &target_name)
{

    // std::vector<std::string> pcd_files;

    pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud;

    // Loop through the directory
    for (const auto &entry : fs::directory_iterator(directory_path))
    {
        if (entry.path().extension() == ".pcd")
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());

            if (pcl::io::loadPCDFile<pcl::PointXYZI>(entry.path().string(), *cloud) == -1)
            {
                std::cerr << "Failed to load PCD file: " << entry.path().string() << std::endl;
                continue;
            }

            // If the filename is source.pcd, set as the source cloud
            if (entry.path().filename() == source_name)
            {
                source_cloud = cloud;
            }
            else if (entry.path().filename() == target_name)
            {
                target_cloud = cloud;
            }
        }
    }

    // Ensure the source cloud was loaded
    if (source_cloud->points.empty())
    {
        throw std::runtime_error("Source PCD  not found in the directory.");
    }
    else if (target_cloud->points.empty())
    {
        throw std::runtime_error("Target PCD  not found in the directory.");
    }

    return {source_cloud, target_cloud};
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lixel_demo");
    ros::NodeHandle nh;
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "/home/gj/catkin_ws_STD/src/STD/logs/";
    FLAGS_alsologtostderr = 1;

    // 从ROS参数服务器获取数据集路径和配置文件路径
    std::string lidar_path = "";
    std::string source_name = "";
    std::string target_name = "";

    nh.param<std::string>("lidar_path", lidar_path, "");   // LiDAR数据集
    nh.param<std::string>("source_name", source_name, ""); // 当前帧名字
    nh.param<std::string>("target_name", target_name, ""); // 当前帧名字

    ConfigSetting config_setting;
    read_parameters(nh, config_setting);
    // 初始化ROS发布器，用于发布点云、位姿和其他可视化信息
    // ros::Publisher pubOdomAftMapped =
    //     nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    // ros::Publisher pubRegisterCloud =
    //     nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
    ros::Publisher pubCureentCloudSource =
        nh.advertise<sensor_msgs::PointCloud2>("/source_cloud_current", 100);
    ros::Publisher pubCureentCloudTarget =
        nh.advertise<sensor_msgs::PointCloud2>("/target_cloud_current", 100);
    ros::Publisher pubCurrentCornerSource =
        nh.advertise<sensor_msgs::PointCloud2>("/source_cloud_key_points", 100);
    ros::Publisher pubCurrentCornerTarget =
        nh.advertise<sensor_msgs::PointCloud2>("/target_cloud_key_points", 100);
    // ros::Publisher pubMatchedCloud =
    //     nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
    // ros::Publisher pubMatchedCorner =
    //     nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
    ros::Publisher pubSTDSource =
        nh.advertise<visualization_msgs::MarkerArray>("source_descriptor_line", 10);
    ros::Publisher pubSTDTarget =
        nh.advertise<visualization_msgs::MarkerArray>("target_descriptor_line", 10);

    ros::Publisher pubPlaneCloudSource =
        nh.advertise<sensor_msgs::PointCloud2>("/source_plane_cloud", 100);
    ros::Publisher pubPlaneCloudTarget =
        nh.advertise<sensor_msgs::PointCloud2>("/target_plane_cloud", 100);

    ros::Rate loop(500);
    ros::Rate slow_loop(10);

    // 读取目录的pcd文件
    auto [source, target] = load_pcds_from_directory(lidar_path, source_name, target_name);
    std::cout << "Loaded source cloud size" << source->points.size() << " and " << target->points.size() << " target clouds." << std::endl;

    // 构造一个描述符管理器用于SLAM的回环检测
    STDescManager *std_manager = new STDescManager(config_setting); // 构造一个描述符管理器

    size_t cloudInd = 0;
    size_t keyCloudInd = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    // for (auto pv : source->points)
    // {
    //     temp_cloud->points.push_back(pv);
    // }

    std::vector<double> descriptor_time;
    std::vector<double> querying_time;
    std::vector<double> update_time;
    int triggle_loop_num = 0;
    // int target_num = target.size();
    int cloud_id = 0;
    while (ros::ok())
    {

        // 将target输入数据库
        if (cloud_id == 0)
        {
            // 创建一个变换矩阵
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            // 定义一个x方向上的平移 (20米)
            transform.translation() << 0.0, 0.0, 15.0;
            pcl::transformPointCloud(*target, *target, transform);

        // 下采样
    LOG(INFO)<<"raw pointcloud size:"<<target->size();
    down_sampling_voxel(*target, config_setting.ds_size_);
    LOG(INFO)<<"down_sampling_voxel size:"<<target->size();

            for (auto pv : target->points)
            {
                temp_cloud->points.push_back(pv);
            }
            LOG(INFO) << "target point cloud id:" << cloud_id
                      << ", cloud size: " << temp_cloud->size();
            // step1. 描述符提取
            auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
            std::vector<STDesc> stds_vec;
            std_manager->GenerateSTDescs(temp_cloud, stds_vec);

            LOG(INFO) << "target point cloud std size:" << stds_vec.size();
            auto t_descriptor_end = std::chrono::high_resolution_clock::now();
            descriptor_time.push_back(time_inc(t_descriptor_end, t_descriptor_begin));

            // step3. 将描述符添加到数据库
            auto t_map_update_begin = std::chrono::high_resolution_clock::now();
            std_manager->AddSTDescs(stds_vec);
            auto t_map_update_end = std::chrono::high_resolution_clock::now();
            update_time.push_back(time_inc(t_map_update_end, t_map_update_begin));

            pcl::PointCloud<pcl::PointXYZI> save_key_cloud;
            save_key_cloud = *temp_cloud;

            std_manager->key_cloud_vec_.push_back(save_key_cloud.makeShared());

            // publish
            // 创建一个用于发布的点云消息
            sensor_msgs::PointCloud2 pub_cloud;
            // 发布temp_cloud
            transform = Eigen::Affine3f::Identity();
            transform.translation() << 0.0, 30.0, 0.0;
            pcl::transformPointCloud(*temp_cloud, *temp_cloud, transform);
            pcl::toROSMsg(*temp_cloud, pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCureentCloudSource.publish(pub_cloud);
            // 发布std_manager中的最新关键点点云
            pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCurrentCornerSource.publish(pub_cloud);
            // 发布平面点云
            pcl::toROSMsg(*std_manager->plane_pointcloud_vec_.back(), pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubPlaneCloudTarget.publish(pub_cloud);

            // 发布与回环匹配的描述符对
            publish_std(stds_vec, pubSTDTarget, "target");
        }
        else if (cloud_id == 1)
        {
                    // 下采样
    LOG(INFO)<<"raw pointcloud size:"<<source->size();
    down_sampling_voxel(*source, config_setting.ds_size_);
    LOG(INFO)<<"down_sampling_voxel size:"<<source->size();
            for (auto pv : source->points)
            {
                temp_cloud->points.push_back(pv);
            }
            LOG(INFO) << "source point cloud id :" << cloud_id
                      << ", cloud size: " << temp_cloud->size();
            // step1. 描述符提取
            auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
            std::vector<STDesc> stds_vec;
            std_manager->GenerateSTDescs(temp_cloud, stds_vec);
            LOG(INFO) << "source point cloud std size:" << stds_vec.size();
            auto t_descriptor_end = std::chrono::high_resolution_clock::now();
            descriptor_time.push_back(time_inc(t_descriptor_end, t_descriptor_begin));

            // step2. 使用描述符搜索回环
            auto t_query_begin = std::chrono::high_resolution_clock::now();
            std::pair<int, double> search_result(-1, 0);
            std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
            loop_transform.first << 0, 0, 0;
            loop_transform.second = Eigen::Matrix3d::Identity();
            std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
            // 进行回环搜索
            std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                    loop_std_pair);
            LOG(INFO) << "loop std pair num :" << loop_std_pair.size();

            // // 如果找到了的回环，则打印信息
            // if (search_result.first > 0)
            // {
            //     std::cout << "[Loop Detection] triggle loop: " << keyCloudInd << "--"
            //               << search_result.first << ", score:" << search_result.second
            //               << std::endl;
            // }
            // auto t_query_end = std::chrono::high_resolution_clock::now();
            // querying_time.push_back(time_inc(t_query_end, t_query_begin));

            // // step3. 将描述符添加到数据库
            // auto t_map_update_begin = std::chrono::high_resolution_clock::now();
            // std_manager->AddSTDescs(stds_vec);
            // auto t_map_update_end = std::chrono::high_resolution_clock::now();
            // update_time.push_back(time_inc(t_map_update_end, t_map_update_begin));
            // std::cout << "[Time] descriptor extraction: "
            //           << time_inc(t_descriptor_end, t_descriptor_begin) << "ms, "
            //           << "query: " << time_inc(t_query_end, t_query_begin) << "ms, "
            //           << "update map:"
            //           << time_inc(t_map_update_end, t_map_update_begin) << "ms"
            //           << std::endl;
            // std::cout << std::endl;

            // pcl::PointCloud<pcl::PointXYZI> save_key_cloud;
            // save_key_cloud = *temp_cloud;

            // std_manager->key_cloud_vec_.push_back(save_key_cloud.makeShared());
            // publish
            // 创建一个用于发布的点云消息
            sensor_msgs::PointCloud2 pub_cloud;
            // 发布temp_cloud
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.translation() << 0.0, 30.0, 0.0;
            pcl::transformPointCloud(*temp_cloud, *temp_cloud, transform);
            pcl::toROSMsg(*temp_cloud, pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCureentCloudTarget.publish(pub_cloud);
            // 发布std_manager中的最新关键点点云
            pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCurrentCornerTarget.publish(pub_cloud);
            // 发布平面点云
            pcl::toROSMsg(*std_manager->plane_pointcloud_vec_.back(), pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubPlaneCloudSource.publish(pub_cloud);
            // 发布与回环匹配的描述符对
            publish_std(stds_vec, pubSTDSource, "source");
        }
        else
        {
        }

        // 如果搜索结果的第一个元素大于0，表示找到了一个可能的回环
        // if (search_result.first > 0)
        // {
        //     // 增加检测到的回环数量
        //     triggle_loop_num++;
        //     // 将检测到的回环的关键帧转换为ROS消息格式
        //     pcl::toROSMsg(*std_manager->key_cloud_vec_[search_result.first],
        //                   pub_cloud);
        //     // 设置消息的坐标系,发布点云
        //     pub_cloud.header.frame_id = "camera_init";
        //     pubMatchedCloud.publish(pub_cloud);

        //     // 短暂休眠，确保数据发布成功
        //     slow_loop.sleep();

        //     // 转换和发布检测到的回环的关键点的点云
        //     pcl::toROSMsg(*std_manager->corner_cloud_vec_[search_result.first],
        //                   pub_cloud);
        //     pub_cloud.header.frame_id = "camera_init";
        //     pubMatchedCorner.publish(pub_cloud);
        //     // 发布与回环匹配的描述符对
        //     publish_std_pairs(loop_std_pair, pubSTD);
        //     // 再次短暂休眠
        //     slow_loop.sleep();
        //     // getchar();
        // }

        temp_cloud->clear();
        cloud_id++;
        loop.sleep();
    }

    google::ShutdownGoogleLogging();
    return 0;
}