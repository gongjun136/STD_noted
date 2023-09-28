#include "include/STDesc.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>

// Read KITTI data
/**
 * @brief 读取KITTI的bin数据集
 *
 * @param lidar_data_path
 * @return std::vector<float>
 */
std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
  std::ifstream lidar_data_file;
  lidar_data_file.open(lidar_data_path,
                       std::ifstream::in | std::ifstream::binary); // 二进制模式打开文件
  // 文件不存在，返回空数组
  if (!lidar_data_file)
  {
    std::cout << "Read End..." << std::endl;
    std::vector<float> nan_data;
    return nan_data;
    // exit(-1);
  }
  // 定位到文件末尾获取数据的元素数量
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  // 从文件中读取LiDAR数据到缓存区
  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
                       num_elements * sizeof(float));
  return lidar_data_buffer;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "demo_kitti");
  ros::NodeHandle nh;
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "/home/gj/catkin_ws_STD/src/STD/logs/";
  FLAGS_alsologtostderr = 1;

  // 从ROS参数服务器获取数据集路径和配置文件路径
  std::string lidar_path = "";
  std::string pose_path = "";
  std::string config_path = "";
  nh.param<std::string>("lidar_path", lidar_path, ""); // LiDAR数据集
  nh.param<std::string>("pose_path", pose_path, "");   // 位姿文件

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);
  // 初始化ROS发布器，用于发布点云、位姿和其他可视化信息
  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  // ros::Publisher pubRegisterCloud =
  //     nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubCureentCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubCurrentCorner =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
  ros::Publisher pubMatchedCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedCorner =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
  ros::Publisher pubSTD =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  ros::Rate loop(500);
  ros::Rate slow_loop(10);

  // 加载位姿和时间戳
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> poses_vec; // 位姿容器
  std::vector<double> times_vec;                                      // 时间戳容器
  load_pose_with_time(pose_path, poses_vec, times_vec);
  std::cout << "Sucessfully load pose with number: " << poses_vec.size()
            << std::endl;
  // 构造一个描述符管理器用于SLAM的回环检测
  STDescManager *std_manager = new STDescManager(config_setting); // 构造一个描述符管理器

  size_t cloudInd = 0;
  size_t keyCloudInd = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZI>());

  std::vector<double> descriptor_time;
  std::vector<double> querying_time;
  std::vector<double> update_time;
  int triggle_loop_num = 0;
  while (ros::ok())
  {
    // 1.读取雷达数据到lidar_data
    std::stringstream lidar_data_path;
    lidar_data_path << lidar_path << std::setfill('0') << std::setw(10)
                    << cloudInd << ".bin";
    std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());

    // 空点云就退出循环
    if (lidar_data.size() == 0)
    {
      break;
    }
    // 2.将点云转到世界系，存放到PCL点云current_cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Vector3d translation = poses_vec[cloudInd].first;
    Eigen::Matrix3d rotation = poses_vec[cloudInd].second;
    for (std::size_t i = 0; i < lidar_data.size(); i += 4)
    {
      pcl::PointXYZI point;
      point.x = lidar_data[i];
      point.y = lidar_data[i + 1];
      point.z = lidar_data[i + 2];
      point.intensity = lidar_data[i + 3];
      Eigen::Vector3d pv = point2vec(point);
      pv = rotation * pv + translation;
      point = vec2point(pv);
      current_cloud->push_back(point);
    }
    // 下采样
    LOG(INFO) << "raw pointcloud size:" << current_cloud->size();
    down_sampling_voxel(*current_cloud, config_setting.ds_size_);
    LOG(INFO) << "down_sampling_voxel size:" << current_cloud->size();
    for (auto pv : current_cloud->points)
    {
      temp_cloud->points.push_back(pv);
    }

    // check if keyframe
    if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0)
    {
      LOG(INFO) << "--------------";
      LOG(INFO) << lidar_data_path.str();
      // LOG(INFO) << "Key Frame id:" << keyCloudInd
      //           << ", cloud size: " << temp_cloud->size() ;
      // step1. 描述符提取
      auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
      std::vector<STDesc> stds_vec;
      std_manager->GenerateSTDescs(temp_cloud, stds_vec);
      auto t_descriptor_end = std::chrono::high_resolution_clock::now();
      descriptor_time.push_back(time_inc(t_descriptor_end, t_descriptor_begin));

      LOG(INFO)<<"STDs num:"<<stds_vec.size();
      // step2. 使用描述符搜索回环
      auto t_query_begin = std::chrono::high_resolution_clock::now();
      std::pair<int, double> search_result(-1, 0);
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
      loop_transform.first << 0, 0, 0;
      loop_transform.second = Eigen::Matrix3d::Identity();
      std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
      // 如果关键帧索引大于预设的跳过数量，则进行回环搜索
      if (keyCloudInd > config_setting.skip_near_num_)
      {
        std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                loop_std_pair);
      }
      // 如果找到了的回环，则打印信息
      if (search_result.first > 0)
      {
        std::cout << "[Loop Detection] triggle loop: " << keyCloudInd << "--"
                  << search_result.first << ", score:" << search_result.second
                  << std::endl;
      }
      auto t_query_end = std::chrono::high_resolution_clock::now();
      querying_time.push_back(time_inc(t_query_end, t_query_begin));

      // step3. 将描述符添加到数据库
      auto t_map_update_begin = std::chrono::high_resolution_clock::now();
      std_manager->AddSTDescs(stds_vec);
      auto t_map_update_end = std::chrono::high_resolution_clock::now();
      update_time.push_back(time_inc(t_map_update_end, t_map_update_begin));
      std::cout << "[Time] descriptor extraction: "
                << time_inc(t_descriptor_end, t_descriptor_begin) << "ms, "
                << "query: " << time_inc(t_query_end, t_query_begin) << "ms, "
                << "update map:"
                << time_inc(t_map_update_end, t_map_update_begin) << "ms"
                << std::endl;
      std::cout << std::endl;

      pcl::PointCloud<pcl::PointXYZI> save_key_cloud;
      save_key_cloud = *temp_cloud;

      std_manager->key_cloud_vec_.push_back(save_key_cloud.makeShared());

      // publish
      // 创建一个用于发布的点云消息
      sensor_msgs::PointCloud2 pub_cloud;
      // 发布temp_cloud
      pcl::toROSMsg(*temp_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCureentCloud.publish(pub_cloud);
      // 发布std_manager中的最新关键点点云
      pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCurrentCorner.publish(pub_cloud);

      // 如果搜索结果的第一个元素大于0，表示找到了一个可能的回环
      if (search_result.first > 0)
      {
        // 增加检测到的回环数量
        triggle_loop_num++;
        // 将检测到的回环的关键帧转换为ROS消息格式
        pcl::toROSMsg(*std_manager->key_cloud_vec_[search_result.first],
                      pub_cloud);
        // 设置消息的坐标系,发布点云
        pub_cloud.header.frame_id = "camera_init";
        pubMatchedCloud.publish(pub_cloud);

        // 短暂休眠，确保数据发布成功
        slow_loop.sleep();

        // 转换和发布检测到的回环的关键点的点云
        pcl::toROSMsg(*std_manager->corner_cloud_vec_[search_result.first],
                      pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubMatchedCorner.publish(pub_cloud);
        // 发布与回环匹配的描述符对
        publish_std_pairs(loop_std_pair, pubSTD);
        // 再次短暂休眠
        slow_loop.sleep();
        // getchar();
      }
      temp_cloud->clear();
      keyCloudInd++;
      loop.sleep();
    }
    // 发布当前帧的位姿
    nav_msgs::Odometry odom;
    odom.header.frame_id = "camera_init";
    odom.pose.pose.position.x = translation[0];
    odom.pose.pose.position.y = translation[1];
    odom.pose.pose.position.z = translation[2];
    Eigen::Quaterniond q(rotation);
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    pubOdomAftMapped.publish(odom);
    loop.sleep();
    cloudInd++;
  }
  double mean_descriptor_time =
      std::accumulate(descriptor_time.begin(), descriptor_time.end(), 0) * 1.0 /
      descriptor_time.size();
  double mean_query_time =
      std::accumulate(querying_time.begin(), querying_time.end(), 0) * 1.0 /
      querying_time.size();
  double mean_update_time =
      std::accumulate(update_time.begin(), update_time.end(), 0) * 1.0 /
      update_time.size();
  std::cout << "Total key frame number:" << keyCloudInd
            << ", loop number:" << triggle_loop_num << std::endl;
  std::cout << "Mean time for descriptor extraction: " << mean_descriptor_time
            << "ms, query: " << mean_query_time
            << "ms, update: " << mean_update_time << "ms, total: "
            << mean_descriptor_time + mean_query_time + mean_update_time << "ms"
            << std::endl;
  google::ShutdownGoogleLogging();
  return 0;
}