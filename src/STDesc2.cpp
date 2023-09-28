#include "include/STDesc2.h"

/**
 * @brief 对点云进行体素下采样
 *
 * @param pl_feat
 * @param voxel_size
 */
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size)
{
  // 随机生成一个0-255之间的强度值，但在该函数内部未使用
  int intensity = rand() % 255;
  // 如果体素大小小于0.01，则不进行下采样并退出
  if (voxel_size < 0.01)
  {
    return;
  }
  // 用于存放体素和对应的点数据的映射
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  // 获取点云的大小
  uint plsize = pl_feat.size();

  // 遍历点云中的每一个点
  for (uint i = 0; i < plsize; i++)
  {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    // 计算点在体素网格中的位置
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    // 创建或更新体素映射
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    // 如果此位置的体素已经存在，更新其数据
    if (iter != voxel_map.end())
    {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    }
    else // 如果体素尚不存在，创建一个新的体素
    {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }

  // 清空原点云并以新的下采样点填充
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
  {
    // 使用体素内所有点的平均值代替原点
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}

/**
 * @brief 读取配置参数
 *
 * @param nh 句柄
 * @param config_setting 配置参数
 */
void read_parameters(ros::NodeHandle &nh, ConfigSetting &config_setting)
{

  // pre-preocess
  nh.param<double>("ds_size", config_setting.ds_size_, 0.5);
  nh.param<int>("maximum_corner_num", config_setting.maximum_corner_num_, 100);

  // key points
  nh.param<double>("plane_merge_normal_thre",
                   config_setting.plane_merge_normal_thre_, 0.1);
  nh.param<double>("plane_detection_thre", config_setting.plane_detection_thre_,
                   0.01);
  nh.param<double>("voxel_size", config_setting.voxel_size_, 2.0);
  nh.param<int>("voxel_init_num", config_setting.voxel_init_num_, 10);
  nh.param<double>("proj_image_resolution",
                   config_setting.proj_image_resolution_, 0.5);
  nh.param<double>("proj_dis_min", config_setting.proj_dis_min_, 0);
  nh.param<double>("proj_dis_max", config_setting.proj_dis_max_, 2);
  nh.param<double>("corner_thre", config_setting.corner_thre_, 10);

  // std descriptor
  nh.param<int>("descriptor_near_num", config_setting.descriptor_near_num_, 10);
  nh.param<double>("descriptor_min_len", config_setting.descriptor_min_len_, 2);
  nh.param<double>("descriptor_max_len", config_setting.descriptor_max_len_,
                   50);
  nh.param<double>("non_max_suppression_radius",
                   config_setting.non_max_suppression_radius_, 2.0);
  nh.param<double>("std_side_resolution", config_setting.std_side_resolution_,
                   0.2);

  // candidate search
  nh.param<int>("skip_near_num", config_setting.skip_near_num_, 50);
  nh.param<int>("candidate_num", config_setting.candidate_num_, 50);
  nh.param<int>("sub_frame_num", config_setting.sub_frame_num_, 10);
  nh.param<double>("rough_dis_threshold", config_setting.rough_dis_threshold_,
                   0.01);
  nh.param<double>("vertex_diff_threshold",
                   config_setting.vertex_diff_threshold_, 0.5);
  nh.param<double>("icp_threshold", config_setting.icp_threshold_, 0.5);
  nh.param<double>("normal_threshold", config_setting.normal_threshold_, 0.2);
  nh.param<double>("dis_threshold", config_setting.dis_threshold_, 0.5);

  std::cout << "Sucessfully load parameters:" << std::endl;
  std::cout << "----------------Main Parameters-------------------"
            << std::endl;
  std::cout << "voxel size:" << config_setting.voxel_size_ << std::endl;
  std::cout << "loop detection threshold: " << config_setting.icp_threshold_
            << std::endl;
  std::cout << "sub-frame number: " << config_setting.sub_frame_num_
            << std::endl;
  std::cout << "candidate number: " << config_setting.candidate_num_
            << std::endl;
  std::cout << "maximum corners size: " << config_setting.maximum_corner_num_
            << std::endl;
}

/**
 * @brief 从pose_file中解析时间戳和位姿信息
 *
 * @param pose_file 文本文件，数据顺序：时间戳、平移xyz、四元数wxyz
 * @param poses_vec
 * @param times_vec
 */
void load_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &poses_vec,
    std::vector<double> &times_vec)
{
  // 1.清空时间戳容器、位姿容器
  times_vec.clear();
  poses_vec.clear();

  // 2.将pose_file所有时间戳给时间戳容器，所有位姿给位姿容器
  std::ifstream fin(pose_file);
  std::string line; // 用于存储每一行内容
  Eigen::Matrix<double, 1, 7> temp_matrix;
  while (getline(fin, line))
  {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ' '))
    {
      if (number == 0)
      {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        times_vec.push_back(time);
        number++;
      }
      else
      {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 7)
        {
          Eigen::Vector3d translation(temp_matrix[0], temp_matrix[1],
                                      temp_matrix[2]);
          Eigen::Quaterniond q(temp_matrix[6], temp_matrix[3], temp_matrix[4],
                               temp_matrix[5]);
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = q.toRotationMatrix();
          poses_vec.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

/**
 * 计算两个时间点之间的差异，并返回该差异的毫秒值。
 *
 * @param t_end 结束的时间点。
 * @param t_begin 开始的时间点。
 * @return 两个时间点之间的差异，以毫秒为单位。
 */
double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin)
{
  // 计算 t_end 和 t_begin 之间的差异，并将其转换为 double 类型的秒数。
  // 然后将秒数乘以 1000，以获取毫秒值。
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end -
                                                                   t_begin)
             .count() *
         1000;
}

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec)
{
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}
Eigen::Vector3d point2vec(const pcl::PointXYZI &pi)
{
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

bool attach_greater_sort(std::pair<double, int> a, std::pair<double, int> b)
{
  return (a.first > b.first);
}

// 辅助函数：设置线段的属性
void setLineProperties(visualization_msgs::Marker &m_line,
                       const std::string &frame_id,
                       const std::string &line_type
                       )
{
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  m_line.scale.x = 0.1;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = frame_id;

  // 根据线段类型设置颜色
  if (line_type == "green")
  {
    m_line.color.r = 138.0 / 255;
    m_line.color.g = 226.0 / 255;
    m_line.color.b = 52.0 / 255;
    m_line.color.a = 0.8;
  }
  else if (line_type == "white")
  {
    m_line.color.r = 1;
    m_line.color.g = 1;
    m_line.color.b = 1;
    m_line.color.a = 0.8;
  }
  else
  {
    m_line.color.a = 0.00;
  }
}

// 辅助函数：连接两个顶点
void connectVertices(visualization_msgs::Marker &m_line,
                     const Eigen::Vector3d &vertex1,
                     const Eigen::Vector3d &vertex2)
{
  geometry_msgs::Point p1, p2;

  p1.x = vertex1[0];
  p1.y = vertex1[1];
  p1.z = vertex1[2];

  p2.x = vertex2[0];
  p2.y = vertex2[1];
  p2.z = vertex2[2];

  m_line.points.push_back(p1);
  m_line.points.push_back(p2);
}

void publish_std(const std::vector<STDesc> &stds,
                 const ros::Publisher &std_publisher,
                 const std::string &pcd_name,
                 const std::string &frame_id)
{
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;

  int max_pub_cnt = 0;

  if (pcd_name == "source")
  {
    for (const auto &std : stds)
    {
      if (max_pub_cnt >= 100)
      {
        break;
      }
      // 使用绿色连接var.second的顶点
      setLineProperties(m_line, frame_id, "green");
      connectVertices(m_line, std.vertex_A_, std.vertex_B_);
      ma_line.markers.push_back(m_line);
      m_line.id++;

      connectVertices(m_line, std.vertex_B_, std.vertex_C_);
      ma_line.markers.push_back(m_line);
      m_line.id++;

      connectVertices(m_line, std.vertex_C_, std.vertex_A_);
      ma_line.markers.push_back(m_line);
      m_line.id++;

      max_pub_cnt++;
    }
  }
  else if (pcd_name == "target")
  {
    for (const auto &std : stds)
    {
      if (max_pub_cnt >= 100)
      {
        break;
      }
      // 使用绿色连接var.second的顶点
      setLineProperties(m_line, frame_id, "white");
      connectVertices(m_line, std.vertex_A_, std.vertex_B_);
      ma_line.markers.push_back(m_line);
      m_line.id++;

      connectVertices(m_line, std.vertex_B_, std.vertex_C_);
      ma_line.markers.push_back(m_line);
      m_line.id++;

      connectVertices(m_line, std.vertex_C_, std.vertex_A_);
      ma_line.markers.push_back(m_line);
      m_line.id++;

      max_pub_cnt++;
    }
  }
  else
  {
    LOG(ERROR) << "pcd_name is error";
    return;
  }
  // 发布线段
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();

  return;
}

// 主函数：发布STD描述符的三个点之间的连接线
void publish_std_pairs(const std::vector<std::pair<STDesc, STDesc>> &match_std_pairs,
                       const ros::Publisher &std_publisher,
                       const std::string &frame_id)
{
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;

  int max_pub_cnt = 0;
  // 遍历每一个STDesc对
  for (const auto &var : match_std_pairs)
  {
    if (max_pub_cnt >= 100)
    {
      break;
    }

    // 使用绿色连接var.second的顶点
    setLineProperties(m_line, frame_id, "green");
    connectVertices(m_line, var.second.vertex_A_, var.second.vertex_B_);
    ma_line.markers.push_back(m_line);
    m_line.id++;

    connectVertices(m_line, var.second.vertex_B_, var.second.vertex_C_);
    ma_line.markers.push_back(m_line);
    m_line.id++;

    connectVertices(m_line, var.second.vertex_C_, var.second.vertex_A_);
    ma_line.markers.push_back(m_line);
    m_line.id++;

    // 使用白色连接var.first的顶点
    setLineProperties(m_line, frame_id, "white");
    connectVertices(m_line, var.first.vertex_A_, var.first.vertex_B_);
    ma_line.markers.push_back(m_line);
    m_line.id++;

    connectVertices(m_line, var.first.vertex_B_, var.first.vertex_C_);
    ma_line.markers.push_back(m_line);
    m_line.id++;

    connectVertices(m_line, var.first.vertex_C_, var.first.vertex_A_);
    ma_line.markers.push_back(m_line);
    m_line.id++;

    max_pub_cnt++;
  }

  // 添加透明线段，确保总数量达到600
  // rviz中，仅仅在代码中清除标记并不会立即在可视化中移除这些标记。
  // 为了在rviz或其他可视化工具中清除或覆盖之前的标记，您需要发送新的透明标记来替代它们
  // visualization_msgs::Marker transparent_line;
  setLineProperties(m_line, frame_id, "transparent");
  for (int i = 0; i < 100 * 6 - ma_line.markers.size(); ++i)
  {
    // Create a dummy point for the transparent line
    geometry_msgs::Point dummy_point;
    dummy_point.x = 0;
    dummy_point.y = 0;
    dummy_point.z = 0;
    m_line.points.push_back(dummy_point);
    m_line.points.push_back(dummy_point);

    ma_line.markers.push_back(m_line);
    m_line.id++;
  }

  // 发布线段
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
  //   for (int j = 0; j < 100 * 6; j++)
  // {
  //   m_line.color.a = 0.00;
  //   ma_line.markers.push_back(m_line);
  //   m_line.id++;
  // }
  // std_publisher.publish(ma_line);
  // m_line.id = 0;
  // ma_line.markers.clear();
}

/**
 * @brief 构建体素地图,提取平面和角点,构建STD描述符
 *
 * @param input_cloud 输入点云
 * @param stds_vec STD描述符
 */
void STDescManager::GenerateSTDescs(
    pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<STDesc> &stds_vec)
{

  // step1, 体素化和平面提取
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map; // 存储每个体素对应的OctoTree
  init_voxel_map(input_cloud, voxel_map);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr plane_pointcloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  getPlane(voxel_map, plane_cloud, plane_pointcloud);
  LOG(INFO) << "[Description] planes size:" << plane_cloud->size();
  plane_cloud_vec_.push_back(plane_cloud);
  plane_pointcloud_vec_.push_back(plane_pointcloud);

  // step2, 建立体素地图中平面之间的联系
  build_connection(voxel_map);

  // step3, 提取角点
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_points(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  corner_extractor(voxel_map, input_cloud, corner_points);
  corner_cloud_vec_.push_back(corner_points);
  // std::cout << "[Description] corners size:" << corner_points->size()
  //           << std::endl;

  // step4, 生成STD
  stds_vec.clear();
  build_stdesc(corner_points, stds_vec);
  // std::cout << "[Description] stds size:" << stds_vec.size() << std::endl;

  // step5, 清理内存
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    delete (iter->second);
  }
  return;
}

/**
 * @brief 搜索闭环
 *
 * @param stds_vec 当前帧的描述符
 * @param loop_result 回环检测结果: (最佳匹配帧的ID, 匹配得分)
 * @param loop_transform  检测到的回环的变换
 * @param loop_std_pair 成功匹配的描述符对
 */
void STDescManager::SearchLoop(
    const std::vector<STDesc> &stds_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<STDesc, STDesc>> &loop_std_pair)
{
  // 检查是否生成描述符
  if (stds_vec.size() == 0)
  {
    LOG(ERROR)<<"No STDescs!";
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
  // step1, 搜索候选者，默认50个
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<STDMatchList> candidate_matcher_vec;
  candidate_selector(stds_vec, candidate_matcher_vec);

  LOG(INFO)<<"候选者数："<<candidate_matcher_vec.size();
  auto t2 = std::chrono::high_resolution_clock::now();

  // step2, 从粗略的候选者中找到最好的候选者，得分为best_score
  double best_score = 0;
  unsigned int best_candidate_id = -1;
  unsigned int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<STDesc, STDesc>> best_sucess_match_vec;
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++)
  {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<STDesc, STDesc>> sucess_match_vec;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,
                     sucess_match_vec);
    if (verify_score > best_score)
    {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  // std::cout << "[Time] candidate selector: " << time_inc(t2, t1)
  //           << " ms, candidate verify: " << time_inc(t3, t2) << "ms"
  //           << std::endl;
  // 结果判断最高得分best_score最否大于阈值
  if (best_score > config_setting_.icp_threshold_)
  {
    // 产生闭环
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    return;
  }
  else
  {
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
}
// Print function for STDesc
void printSTDesc(const STDesc& desc) {
    LOG(INFO) << "Side Length: " << desc.side_length_.transpose() ;
    LOG(INFO) << "Angle: " << desc.angle_.transpose() ;
    LOG(INFO) << "Center: " << desc.center_.transpose() ;
    LOG(INFO) << "Frame ID: " << desc.frame_id_ ;
    LOG(INFO) << "Vertex A: " << desc.vertex_A_.transpose() ;
    LOG(INFO) << "Vertex B: " << desc.vertex_B_.transpose() ;
    LOG(INFO) << "Vertex C: " << desc.vertex_C_.transpose() ;
    LOG(INFO) << "Vertex Attached: " << desc.vertex_attached_.transpose() ;
}
// Function to print the database content
void printDatabase(const std::unordered_map<STDesc_LOC, std::vector<STDesc>>& data_base_) {
    for (const auto& pair : data_base_) {
        const STDesc_LOC& key = pair.first;
        const std::vector<STDesc>& values = pair.second;

        LOG(INFO) << "Key [x: " << key.x << ", y: " << key.y << ", z: " << key.z << "]";
        LOG(INFO) << "Values:";
        for (const auto& value : values) {
            printSTDesc(value);
            LOG(INFO)<< "-----";
        }
        LOG(INFO) << "========\n";
    }
}

void STDescManager::AddSTDescs(const std::vector<STDesc> &stds_vec)
{
  // update frame id
  // 更新帧id
  current_frame_id_++;
  // 遍历STD数组，将所有STD添加到数据库
  for (auto single_std : stds_vec)
  {
    // calculate the position of single std
    STDesc_LOC position;
    // (int)的操作是将相似的position添加到同一容器
    position.x = (int)(single_std.side_length_[0] + 0.5);
    position.y = (int)(single_std.side_length_[1] + 0.5);
    position.z = (int)(single_std.side_length_[2] + 0.5);
    position.a = (int)(single_std.angle_[0]);
    position.b = (int)(single_std.angle_[1]);
    position.c = (int)(single_std.angle_[2]);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end())
    {
      data_base_[position].push_back(single_std);
    }
    else
    {
      std::vector<STDesc> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }

  }
  // printDatabase(data_base_);
  return;
}

/**
 * @brief 初始化体素地图
 *
 * @param input_cloud 输入点云
 * @param voxel_map 用于存储体素数据的体素地图
 */
void STDescManager::init_voxel_map(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map)
{
  // 遍历输入点云
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++)
  {
    // 为每个点计算其在体素地图中的位置
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }

    // 为该点创建一个体素位置
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    // 检查这个体素位置是否已经在体素地图中
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end())
    {
      // 如果存在，则将当前点添加到该位置的体素中
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
    else
    {
      // 如果不存在，则创建一个新的OctoTree，并将当前点添加到该体素中
      OctoTree *octo_tree = new OctoTree(config_setting_);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }
  // 遍历体素地图，初始化每个体素中的OctoTree
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
  {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
  }
  // speed up initialization
  // #ifdef MP_EN
  //   omp_set_num_threads(MP_PROC_NUM);
  //   std::cout << "omp num:" << MP_PROC_NUM << std::endl;
  // #pragma omp parallel for
  // #endif
  // 遍历索引列表，为每个OctoTree进行初始化
  for (int i = 0; i < index.size(); i++)
  {
    iter_list[i]->second->init_octo_tree();
  }
  // std::cout << "voxel num:" << index.size() << std::endl;
  // std::for_each(
  //     std::execution::par_unseq, index.begin(), index.end(),
  //     [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
}

/**
 * @brief 在体素地图中建立平面之间的连接关系,判断哪些平面体素是相邻的并可以被合并连接。
 *
 * @param voxel_map
 */
void STDescManager::build_connection(
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map)
{
  // 遍历体素
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    // 检查当前体素是否被标记为平面
    if (iter->second->plane_ptr_->is_plane_)
    {
      OctoTree *current_octo = iter->second;
      // 对于每个体素，检查其在6个方向上的邻居
      for (int i = 0; i < 6; i++)
      {
        // 根据i的值确定邻居的方向
        VOXEL_LOC neighbor = iter->first;
        if (i == 0)
        {
          neighbor.x = neighbor.x + 1;
        }
        else if (i == 1)
        {
          neighbor.y = neighbor.y + 1;
        }
        else if (i == 2)
        {
          neighbor.z = neighbor.z + 1;
        }
        else if (i == 3)
        {
          neighbor.x = neighbor.x - 1;
        }
        else if (i == 4)
        {
          neighbor.y = neighbor.y - 1;
        }
        else if (i == 5)
        {
          neighbor.z = neighbor.z - 1;
        }
        // 检查邻居是否存在于体素地图中
        auto near = voxel_map.find(neighbor);
        if (near == voxel_map.end())
        {
          // 如果邻居不存在，标记为不连接
          current_octo->is_check_connect_[i] = true;
          current_octo->connect_[i] = false;
        }
        else
        {
          // 如果邻居存在，检查是否已经检查过连接
          if (!current_octo->is_check_connect_[i])
          {
            // 没有检查过连接，判断是否可以连接
            OctoTree *near_octo = near->second;
            current_octo->is_check_connect_[i] = true;
            int j;
            if (i >= 3)
            {
              j = i - 3;
            }
            else
            {
              j = i + 3;
            }
            near_octo->is_check_connect_[j] = true;
            // 判断邻居是否也被标记为平面
            if (near_octo->plane_ptr_->is_plane_)
            {
              // 如果邻居也是平面，计算当前平面与邻居平面的法线差异
              Eigen::Vector3d normal_diff = current_octo->plane_ptr_->normal_ -
                                            near_octo->plane_ptr_->normal_;
              Eigen::Vector3d normal_add = current_octo->plane_ptr_->normal_ +
                                           near_octo->plane_ptr_->normal_;
              // 根据法线之差或之和的范数判断是否连接
              if (normal_diff.norm() <
                      config_setting_.plane_merge_normal_thre_ ||
                  normal_add.norm() <
                      config_setting_.plane_merge_normal_thre_)
              {
                // 如果法线之差或之和的范数小于给定的阈值，将两个平面连接
                current_octo->connect_[i] = true;
                near_octo->connect_[j] = true;
                current_octo->connect_tree_[i] = near_octo;
                near_octo->connect_tree_[j] = current_octo;
              }
              else
              {
                // 如果法线之差或之和的范数不小于给定的阈值，不连接
                current_octo->connect_[i] = false;
                near_octo->connect_[j] = false;
              }
            }
            else
            {
              // 如果邻居不是平面，标记为不连接
              current_octo->connect_[i] = false;
              near_octo->connect_[j] = true;
              near_octo->connect_tree_[j] = current_octo;
            }
          }
        }
      }
    }
  }
}

/**
 * @brief 提取体素地图中的平面，并将它们保存到点云中。
 *
 * @param voxel_map 输入的体素地图，其中每个体素可能包含平面信息。
 * @param plane_cloud 输出点云，其中每个点代表一个平面的中心点，其法线表示平面的方向。
 */
void STDescManager::getPlane(
    const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &plane_pointcloud)
{
  // pcl::PointCloud<pcl::PointXYZI>::Ptr plane_pointcloud(new pcl::PointCloud<pcl::PointXYZI>);

  // 遍历提供的体素地图
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    // 检查当前体素是否包含一个平面
    if (iter->second->plane_ptr_->is_plane_)
    {
      // 如果包含平面，创建一个新的点来代表这个平面
      pcl::PointXYZINormal pi;
      // 使用平面的中心作为点的位置
      pi.x = iter->second->plane_ptr_->center_[0];
      pi.y = iter->second->plane_ptr_->center_[1];
      pi.z = iter->second->plane_ptr_->center_[2];
      // 使用平面的法线作为点的法线
      pi.normal_x = iter->second->plane_ptr_->normal_[0];
      pi.normal_y = iter->second->plane_ptr_->normal_[1];
      pi.normal_z = iter->second->plane_ptr_->normal_[2];
      // 将此点添加到输出点云中
      plane_cloud->push_back(pi);

      // 提取当前体素的所有点并将其添加到plane_cloud中
      const std::vector<Eigen::Vector3d> &voxel_points = iter->second->voxel_points_;
      for (const Eigen::Vector3d &point : voxel_points)
      {
        pcl::PointXYZI voxel_point;
        voxel_point.x = static_cast<float>(point[0]);
        voxel_point.y = static_cast<float>(point[1]);
        voxel_point.z = static_cast<float>(point[2]);
        plane_pointcloud->push_back(voxel_point);
      }
    }
  }
}

/**
 * @brief 通过检查每个非平面体素及其与平面体素的相邻关系，来确定投影点并执行2D投影和角点提取。
 *
 * @param voxel_map 输入的体素地图，其中每个体素可能包含平面信息。
 * @param input_cloud 输入的点云
 * @param corner_points 输出的角点。
 */
void STDescManager::corner_extractor(
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points)
{
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_corner_points(
      new pcl::PointCloud<pcl::PointXYZINormal>);

  // 为了避免因不同视角导致的不一致的体素切割，定义一个向量来存储邻居体素的坐标
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++)
  {
    for (int y = -1; y <= 1; y++)
    {
      for (int z = -1; z <= 1; z++)
      {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  // 遍历所有体素
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    // 判断是否设置为平面
    if (!iter->second->plane_ptr_->is_plane_)
    {
      // 如果当前体素不是平面
      VOXEL_LOC current_position = iter->first;
      OctoTree *current_octo = iter->second;
      int connect_index = -1;
      // 检查当前体素的6个方向
      for (int i = 0; i < 6; i++)
      {
        if (current_octo->connect_[i])
        {
          // 如果当前体素在指定方向上与另一个平面体素连接
          connect_index = i;
          OctoTree *connect_octo = current_octo->connect_tree_[connect_index];
          bool use = false;
          // 检查与当前体素连接的平面体素的6个方向
          for (int j = 0; j < 6; j++)
          {
            if (connect_octo->is_check_connect_[j])
            {
              if (connect_octo->connect_[j])
              {
                use = true;
              }
            }
          }
          // 如果连接的平面体素附近没有其他平面体素，跳过此次循环
          if (use == false)
          {
            continue;
          }
          // only project voxels with points num > 10
          // 如果当前体素中的点数大于10
          if (current_octo->voxel_points_.size() > 10)
          {
            Eigen::Vector3d projection_normal =
                current_octo->connect_tree_[connect_index]->plane_ptr_->normal_;
            Eigen::Vector3d projection_center =
                current_octo->connect_tree_[connect_index]->plane_ptr_->center_;
            std::vector<Eigen::Vector3d> proj_points;
            // 遍历邻居体素坐标向量
            for (auto voxel_inc : voxel_round)
            {
              // 根据当前体素的位置和增量计算邻居体素的位置
              VOXEL_LOC connect_project_position = current_position;
              connect_project_position.x += voxel_inc[0];
              connect_project_position.y += voxel_inc[1];
              connect_project_position.z += voxel_inc[2];
              // 在体素地图中查找该邻居体素
              auto iter_near = voxel_map.find(connect_project_position);
              // 如果在体素地图中找到邻居体素
              if (iter_near != voxel_map.end())
              {
                // 初始化标志，用于决定是否跳过当前邻居体素
                bool skip_flag = false;
                // 检查该邻居体素是否是平面
                if (!voxel_map[connect_project_position]
                         ->plane_ptr_->is_plane_)
                {
                  // 如果不是平面
                  // 判断是否投影
                  if (voxel_map[connect_project_position]->is_project_)
                  {
                    // 如果邻居体素之前进行过投影
                    // 检查之前的投影是否与当前的投影法线接近
                    for (auto normal : voxel_map[connect_project_position]
                                           ->proj_normal_vec_)
                    {
                      Eigen::Vector3d normal_diff = projection_normal - normal;
                      Eigen::Vector3d normal_add = projection_normal + normal;
                      // 如果之前的投影法线与当前的投影法线相似，则设置跳过标志
                      if (normal_diff.norm() < 0.5 || normal_add.norm() < 0.5)
                      {
                        skip_flag = true;
                      }
                    }
                  }
                  // 如果设置了跳过标志，则跳过当前邻居体素
                  if (skip_flag)
                  {
                    continue;
                  }
                  // 否则，将该邻居体素中的点添加到投影点云中，并更新其投影状态
                  for (size_t j = 0; j < voxel_map[connect_project_position]
                                             ->voxel_points_.size();
                       j++)
                  {
                    proj_points.push_back(
                        voxel_map[connect_project_position]->voxel_points_[j]);
                    voxel_map[connect_project_position]->is_project_ = true;
                    voxel_map[connect_project_position]
                        ->proj_normal_vec_.push_back(projection_normal);
                  }
                }
              }
            }

            // 执行2D投影并提取角点
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr sub_corner_points(
                new pcl::PointCloud<pcl::PointXYZINormal>);
            extract_corner(projection_center, projection_normal, proj_points,
                           sub_corner_points);
            // 将角点存储到prepare_corner_points
            for (auto pi : sub_corner_points->points)
            {
              prepare_corner_points->push_back(pi);
            }
          }
        }
      }
    }
  }
  // 非极大值抑制
  non_maxi_suppression(prepare_corner_points);
  // 检查是否需要进一步筛选角点
  if (config_setting_.maximum_corner_num_ > prepare_corner_points->size())
  {
    // 不大于最大角点数目，则不需要再处理
    corner_points = prepare_corner_points;
  }
  else
  {
    // 如果角点过多，根据角点的强度值进行排序并选择最强的角点
    std::vector<std::pair<double, int>> attach_vec;
    for (size_t i = 0; i < prepare_corner_points->size(); i++)
    {
      attach_vec.push_back(std::pair<double, int>(
          prepare_corner_points->points[i].intensity, i));
    }
    std::sort(attach_vec.begin(), attach_vec.end(), attach_greater_sort);
    for (size_t i = 0; i < config_setting_.maximum_corner_num_; i++)
    {
      corner_points->points.push_back(
          prepare_corner_points->points[attach_vec[i].second]);
    }
  }
}

/**
 * @brief 2D投影并角点提取
 *
 * @param proj_center
 * @param proj_normal 投影平面法向量
 * @param proj_points 输入的投影点点云
 * @param corner_points 提取到的角点
 */
void STDescManager::extract_corner(
    const Eigen::Vector3d &proj_center, const Eigen::Vector3d proj_normal,
    const std::vector<Eigen::Vector3d> proj_points,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points)
{
  // 1. 计算投影平面的两个正交轴x_axis和y_axis
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double A = proj_normal[0];
  double B = proj_normal[1];
  double C = proj_normal[2];
  double D = -(A * proj_center[0] + B * proj_center[1] + C * proj_center[2]);

  Eigen::Vector3d x_axis(1, 1, 0);
  if (C != 0)
  {
    x_axis[2] = -(A + B) / C;
  }
  else if (B != 0)
  {
    x_axis[1] = -A / B;
  }
  else
  {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = proj_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx =
      -(ax * proj_center[0] + bx * proj_center[1] + cx * proj_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy =
      -(ay * proj_center[0] + by * proj_center[1] + cy * proj_center[2]);
  // 2. 遍历输入的3D点proj_points
  std::vector<Eigen::Vector2d> point_list_2d;
  for (size_t i = 0; i < proj_points.size(); i++)
  {
    // 检查3D点与投影平面的距离是否在指定的距离阈值范围内
    double x = proj_points[i][0];
    double y = proj_points[i][1];
    double z = proj_points[i][2];
    double dis = fabs(x * A + y * B + z * C + D);
    if (dis < dis_threshold_min || dis > dis_threshold_max)
    {
      continue;
    }
    // 将3D点投影到平面并转换为2D点
    Eigen::Vector3d cur_project;
    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
  }
  // 3. 根据2D点的集合确定2D空间的边界
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5)
  {
    return;
  }
  for (auto pi : point_list_2d)
  {
    if (pi[0] < min_x)
    {
      min_x = pi[0];
    }
    if (pi[0] > max_x)
    {
      max_x = pi[0];
    }
    if (pi[1] < min_y)
    {
      min_y = pi[1];
    }
    if (pi[1] > max_y)
    {
      max_y = pi[1];
    }
  }
  // 4. 对2D空间进行xy分段
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  // 初始化存储分段信息的数组
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);
  std::vector<Eigen::Vector2d> img_container[x_axis_len][y_axis_len];
  double img_count_array[x_axis_len][y_axis_len] = {0};
  double gradient_array[x_axis_len][y_axis_len] = {0};
  double mean_x_array[x_axis_len][y_axis_len] = {0};
  double mean_y_array[x_axis_len][y_axis_len] = {0};
  for (int x = 0; x < x_axis_len; x++)
  {
    for (int y = 0; y < y_axis_len; y++)
    {
      img_count_array[x][y] = 0;
      mean_x_array[x][y] = 0;
      mean_y_array[x][y] = 0;
      gradient_array[x][y] = 0;
      std::vector<Eigen::Vector2d> single_container;
      img_container[x][y] = single_container;
    }
  }
  // 5. 根据2D点的分布计算梯度
  // 计算一个小区域内的点数与其邻近区域的点数的差异，这个差异可以用来判断当前区域是否为一个角点
  for (size_t i = 0; i < point_list_2d.size(); i++)
  {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_array[x_index][y_index] += point_list_2d[i][0];
    mean_y_array[x_index][y_index] += point_list_2d[i][1];
    img_count_array[x_index][y_index]++;
    img_container[x_index][y_index].push_back(point_list_2d[i]);
  }
  // 梯度计算，根据分段与相邻分段的点云差计算梯度值，此梯度值可用于识别角点
  for (int x = 0; x < x_axis_len; x++)
  {
    for (int y = 0; y < y_axis_len; y++)
    {
      double gradient = 0; // 用于存储当前位置的梯度
      int cnt = 0;         // 记录邻近区域的数量
      int inc = 1;         // 定义一个搜索的半径，这里为1，表示只考虑直接的邻居
      // 遍历当前位置的邻居
      for (int x_inc = -inc; x_inc <= inc; x_inc++)
      {
        for (int y_inc = -inc; y_inc <= inc; y_inc++)
        {
          int xx = x + x_inc;
          int yy = y + y_inc;
          // 确保新位置在定义的边界内
          if (xx >= 0 && xx < x_axis_len && yy >= 0 && yy < y_axis_len)
          {
            // 排除当前位置，因为我们只对邻居感兴趣
            if (xx != x || yy != y)
            {
              // 如果邻居位置的点数不为0
              if (img_count_array[xx][yy] >= 0)
              {
                // 计算当前位置与邻居位置点数的差异，并累加到gradient上
                gradient += img_count_array[x][y] - img_count_array[xx][yy];
                // 增加邻近区域的计数
                cnt++;
              }
            }
          }
        }
      }
      // 计算平均梯度
      if (cnt != 0)
      {
        gradient_array[x][y] = gradient * 1.0 / cnt;
      }
      else
      {
        gradient_array[x][y] = 0;
      }
    }
  }
  // 6. 根据梯度提取角点
  // 初始化三个向量来存储：最大的梯度值、对应的x轴索引、对应的y轴索引
  std::vector<int> max_gradient_vec;
  std::vector<int> max_gradient_x_index_vec;
  std::vector<int> max_gradient_y_index_vec;
  // 遍历所有分段
  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++)
  {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++)
    {
      // 初始化当前分段的最大梯度及其索引
      double max_gradient = 0;
      int max_gradient_x_index = -10;
      int max_gradient_y_index = -10;
      // 在当前分段内查找最大梯度及其索引
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++)
      {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++)
        {
          if (img_count_array[x_index][y_index] > max_gradient)
          {
            max_gradient = img_count_array[x_index][y_index]; // 更新最大梯度
            max_gradient_x_index = x_index;                   // 更新x索引
            max_gradient_y_index = y_index;                   // 更新y索引
          }
        }
      }
      // 如果最大梯度大于预定义的阈值，则认为这是一个角点
      if (max_gradient >= config_setting_.corner_thre_)
      {
        max_gradient_vec.push_back(max_gradient);                 // 存储最大梯度
        max_gradient_x_index_vec.push_back(max_gradient_x_index); // 存储x索引
        max_gradient_y_index_vec.push_back(max_gradient_y_index); // 存储y索引
      }
    }
  }
  // 7. 确定是否是角点，并重构3D角点添加到corner_points中
  // 初始化一个方向列表，它定义了4个基本的2D方向：上、下、左、右。
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1); // 向上
  direction_list.push_back(d);
  d << 1, 0; // 向右
  direction_list.push_back(d);
  d << 1, 1; // 右上对角
  direction_list.push_back(d);
  d << 1, -1; // 右下对角
  direction_list.push_back(d);
  // 遍历之前根据梯度提取出来的可能的角点
  for (size_t i = 0; i < max_gradient_vec.size(); i++)
  {
    // 标志变量，初始设为true，表示可以添加这个角点
    bool is_add = true;
    for (int j = 0; j < 4; j++)
    {
      // 获取当前点的坐标（x 和 y 分段索引）
      Eigen::Vector2i p(max_gradient_x_index_vec[i],
                        max_gradient_y_index_vec[i]);
      // 计算在给定方向上的两个点的坐标
      Eigen::Vector2i p1 = p + direction_list[j];
      Eigen::Vector2i p2 = p - direction_list[j];
      // 设定一个阈值，这里是当前点的点数的一半
      int threshold = img_count_array[p[0]][p[1]] / 2;
      // 如果在给定方向上的两个点的点数都大于阈值，则继续，否则将is_add设为false
      if (img_count_array[p1[0]][p1[1]] >= threshold &&
          img_count_array[p2[0]][p2[1]] >= threshold)
      {
        // is_add = false;
      }
      else
      {
        continue;
      }
    }
    // 如果is_add为true，说明当前点是一个角点
    if (is_add)
    {
      // 根据2D坐标（即分段的坐标）和之前计算的x_axis和y_axis以及投影中心重建3D坐标
      double px = mean_x_array[max_gradient_x_index_vec[i]]
                              [max_gradient_y_index_vec[i]] /
                  img_count_array[max_gradient_x_index_vec[i]]
                                 [max_gradient_y_index_vec[i]];
      double py = mean_y_array[max_gradient_x_index_vec[i]]
                              [max_gradient_y_index_vec[i]] /
                  img_count_array[max_gradient_x_index_vec[i]]
                                 [max_gradient_y_index_vec[i]];
      // 给定一个3D平面，我们可以使用两个正交的方向向量（x_axis 和 y_axis）来表示该平面上的任意点。
      // px 和 py 提供了这两个方向上的缩放因子，而 proj_center 为我们提供了一个参考点（平面上的一个已知点）。
      // 通过组合这些信息，我们可以从2D坐标恢复3D空间中的点。
      // 这个3D点仍然是平面上的一点，只不过参考系变成了世界系，所以z轴有了值
      Eigen::Vector3d coord = py * x_axis + px * y_axis + proj_center;
      // 创建一个新的3D点，并设置其坐标、强度和法线
      pcl::PointXYZINormal pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      pi.intensity = max_gradient_vec[i];
      pi.normal_x = proj_normal[0];
      pi.normal_y = proj_normal[1];
      pi.normal_z = proj_normal[2];
      // 将新的3D点添加到角点点云中
      corner_points->points.push_back(pi);
    }
  }
  return;
}

/**
 * @brief 实现非极大值抑制，确保只有最显著的角点被保留。
 * 这是通过比较每个角点和它周围的角点的强度来实现的。如果一个角点的强度小于它周围的任何一个角点的强度，它就会被抑制（即不被保留）
 * @param corner_points
 */
void STDescManager::non_maxi_suppression(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points)
{
  // 复制输入的角点到一个新的点云prepare_key_cloud
  std::vector<bool> is_add_vec; // 标志向量，用于记录角点是否应该被保留
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  // 使用kd树来进行半径搜索
  pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree;
  for (auto pi : corner_points->points)
  {
    prepare_key_cloud->push_back(pi);
    is_add_vec.push_back(true);
  }
  // 设置kd树的输入点云
  kd_tree.setInputCloud(prepare_key_cloud);
  // 非极大值抑制过程
  // 遍历每一个角点，查找其一定半径内的其他角点
  // 如果当前角点的强度小于任何邻居角点的强度，则将其标记为不保留
  // 强度较高的点通常更加稳定
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  double radius = config_setting_.non_max_suppression_radius_;
  for (size_t i = 0; i < prepare_key_cloud->size(); i++)
  {
    pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i];
    if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                             pointRadiusSquaredDistance) > 0)
    {
      Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
      {
        Eigen::Vector3d pj(
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].x,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].y,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
        if (pointIdxRadiusSearch[j] == i)
        {
          continue;
        }
        if (prepare_key_cloud->points[i].intensity <=
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].intensity)
        {
          is_add_vec[i] = false; // 标记当前点不保留
        }
      }
    }
  }
  // 使用标志向量筛选并保留角点
  corner_points->clear();
  for (size_t i = 0; i < is_add_vec.size(); i++)
  {
    if (is_add_vec[i])
    {
      corner_points->points.push_back(prepare_key_cloud->points[i]);
    }
  }
  return;
}

/**
 * @brief 给定的角点构建空间三角形描述符（STDesc）
 *
 * @param corner_points 角点
 * @param stds_vec STD描述符
 */
void STDescManager::build_stdesc(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
    std::vector<STDesc> &stds_vec)
{
  // 1. 初始化描述符数组
  stds_vec.clear();
  // 2. 创建KD树，用于角点corner_points的最近邻搜索
  double scale = 1.0 / config_setting_.std_side_resolution_;
  int near_num = config_setting_.descriptor_near_num_;
  double max_dis_threshold = config_setting_.descriptor_max_len_;
  double min_dis_threshold = config_setting_.descriptor_min_len_;
  // 使用哈希映射存储已生成的描述符，确保描述符的唯一性
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  // 创建KD树，用于快速查找点云中的最近邻点
  pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
  kd_tree->setInputCloud(corner_points);
  std::vector<int> pointIdxNKNSearch(near_num);         // 存储最近邻点的索引
  std::vector<float> pointNKNSquaredDistance(near_num); // 存储到最近邻点的平方距离
  // Search N nearest corner points to form stds.
  // 3. 遍历输入的角点
  for (size_t i = 0; i < corner_points->size(); i++)
  {
    // 对于每个角点，查找其最近的`near_num`个邻居
    pcl::PointXYZINormal searchPoint = corner_points->points[i];
    if (kd_tree->nearestKSearch(searchPoint, near_num, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0)
    {
      // 对找到的邻居进行两两组合，尝试形成三角形描述符
      for (int m = 1; m < near_num - 1; m++)
      {
        for (int n = m + 1; n < near_num; n++)
        {
          // 3.1. 计算三角形的三条边的长度
          pcl::PointXYZINormal p1 = searchPoint;
          pcl::PointXYZINormal p2 = corner_points->points[pointIdxNKNSearch[m]];
          pcl::PointXYZINormal p3 = corner_points->points[pointIdxNKNSearch[n]];
          Eigen::Vector3d normal_inc1(p1.normal_x - p2.normal_x,
                                      p1.normal_y - p2.normal_y,
                                      p1.normal_z - p2.normal_z);
          Eigen::Vector3d normal_inc2(p3.normal_x - p2.normal_x,
                                      p3.normal_y - p2.normal_y,
                                      p3.normal_z - p2.normal_z);
          Eigen::Vector3d normal_add1(p1.normal_x + p2.normal_x,
                                      p1.normal_y + p2.normal_y,
                                      p1.normal_z + p2.normal_z);
          Eigen::Vector3d normal_add2(p3.normal_x + p2.normal_x,
                                      p3.normal_y + p2.normal_y,
                                      p3.normal_z + p2.normal_z);
          // 计算边长a、b、c
          // a 是 p1 和 p2 之间的距离。
          // b 是 p1 和 p3 之间的距离。
          // c 是 p2 和 p3 之间的距离。
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          // 判断边长是否在预设的范围内
          if (a > max_dis_threshold || b > max_dis_threshold ||
              c > max_dis_threshold || a < min_dis_threshold ||
              b < min_dis_threshold || c < min_dis_threshold)
          {
            continue;
          }
          // 3.2. 对三角形的边进行排序，确保 a <= b <= c
          double temp;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          // 初始化l1、l2和l3
          // 下标0表示的是a边，相同的是l1,l2,表示p1和p2连接的边；
          // 下标1表示的是b边，相同的是l1,l3，表示p1,p3点连接的边；
          // 下标2表示的是c边，相同的是l2和l3，表示p2和p3连接的边。
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b)
          {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c)
          {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
            if (a > b)
            {
              temp = a;
              a = b;
              b = temp;
              l_temp = l1;
              l1 = l2;
              l2 = l_temp;
            }
          }

          // 3.4. 描述符增强，并生成在feat_map中
          // 根据边长构建一个描述符的唯一键
          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          // 检查该描述符的键是否已经存在于feat_map中
          auto iter = feat_map.find(position);
          // 初始化三角形的三个顶点和对应的法线
          Eigen::Vector3d A, B, C;
          Eigen::Vector3d normal_1, normal_2, normal_3;
          // 如果该描述符键不在feat_map中，则进一步处理
          if (iter == feat_map.end())
          {

            Eigen::Vector3d vertex_attached;
            // 1)判断最长边和最长边相对的点，赋值给点A
            if (l1[0] == l2[0])
            {
              // l1[0]与l2[0]相等表示边c是最长边,p1是与最长边相对的点
              A << p1.x, p1.y, p1.z;
              normal_1 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[0] = p1.intensity;
            }
            else if (l1[1] == l2[1])
            {
              // l1[1]与l2[1]相等表示b是最长边，p2是与最长边相对的点
              A << p2.x, p2.y, p2.z;
              normal_1 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[0] = p2.intensity;
            }
            else
            {
              // l1[2]与l2[2]相等表示a是最长边，p3是与最长边相对的点
              A << p3.x, p3.y, p3.z;
              normal_1 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[0] = p3.intensity;
            }
            // 2)判断次长边和次长边相对的点，赋值给点B
            if (l1[0] == l3[0])
            {
              // l1[0]与l3[0]相等表示c是次长边，p1是次长边相对的点
              B << p1.x, p1.y, p1.z;
              normal_2 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[1] = p1.intensity;
            }
            else if (l1[1] == l3[1])
            {
              // l1[1]与l3[1]相等表示b是次长边，p2是次长边相对的点
              B << p2.x, p2.y, p2.z;
              normal_2 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[1] = p2.intensity;
            }
            else
            {
              // l1[2]与l3[2]相等表示a是次长边，p3是次长边相对的点
              B << p3.x, p3.y, p3.z;
              normal_2 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[1] = p3.intensity;
            }
            // 3)判断最短边和最短边相对的点，赋值给点C
            if (l2[0] == l3[0])
            {
              // l2[0]与l3[0]相等表示c是最短边，p1是最短边相对的点
              C << p1.x, p1.y, p1.z;
              normal_3 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[2] = p1.intensity;
            }
            else if (l2[1] == l3[1])
            {
              // l2[1]与l3[1]相等表示b是最短边，p2是最短边相对的点
              C << p2.x, p2.y, p2.z;
              normal_3 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[2] = p2.intensity;
            }
            else
            {
              // l2[2]与l3[2]相等表示a是最短边，p3是最短边相对的点
              C << p3.x, p3.y, p3.z;
              normal_3 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[2] = p3.intensity;
            }
            // 使用上述的信息构建描述符
            // 描述符包括三角形的三个顶点的位置、法线和强度信息、边长以及法线之间的角度
            STDesc single_descriptor;
            single_descriptor.vertex_A_ = A;
            single_descriptor.vertex_B_ = B;
            single_descriptor.vertex_C_ = C;
            single_descriptor.center_ = (A + B + C) / 3;                       // 计算三角形的中心点
            single_descriptor.vertex_attached_ = vertex_attached;              // 保存强度信息
            single_descriptor.side_length_ << scale * a, scale * b, scale * c; // 保存缩放后的边长
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));    // 计算法线之间的角度
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_id_ = current_frame_id_; // 保存当前帧的ID
            Eigen::Matrix3d triangle_positon;
            feat_map[position] = true;
            // 将该描述符添加到描述符向量中，以供后续使用
            stds_vec.push_back(single_descriptor);
          }
        }
      }
    }
  }
};

/**
 * @brief 从给定的描述符集合中选择回环的候选匹配
 *
 * @param stds_vec 当前帧的描述符
 * @param candidate_matcher_vec 输出的候选匹配列表
 */
void STDescManager::candidate_selector(
    const std::vector<STDesc> &stds_vec,
    std::vector<STDMatchList> &candidate_matcher_vec)
{
  // 1. 初始化相关变量
  double match_array[MAX_FRAME_N] = {0};            // 匹配得分数组
  std::vector<std::pair<STDesc, STDesc>> match_vec; // 存储匹配对
  std::vector<int> match_index_vec;                 // 匹配对的索引

  // 定义3x3x3的体素，用于构建附近的相对位置
  std::vector<Eigen::Vector3i> voxel_round; // 存储周围体素的偏移，用于构建相对位置
  for (int x = -1; x <= 1; x++)
  {
    for (int y = -1; y <= 1; y++)
    {
      for (int z = -1; z <= 1; z++)
      {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }

  // 初始化匹配相关的数据结构
  std::vector<bool> useful_match(stds_vec.size());
  std::vector<std::vector<size_t>> useful_match_index(stds_vec.size());
  std::vector<std::vector<STDesc_LOC>> useful_match_position(stds_vec.size());
  std::vector<size_t> index(stds_vec.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_match[i] = false;
  }
  // 2. 并行处理每个描述符，进行加速匹配
  // 使用OpenMP来进行并行处理，可以大大加快匹配速度，特别是当描述符数量很大时。
  int dis_match_cnt = 0;   // 计数器，用于统计满足距离阈值条件的匹配对数量
  int final_match_cnt = 0; // 计数器，用于统计最终满足所有条件的匹配对数量
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM); // 设置并行线程的数量
#pragma omp parallel for
#endif

  // 遍历每一个描述符，检查其与数据库中的描述符的匹配情况
  for (size_t i = 0; i < stds_vec.size(); i++)
  {
    STDesc src_std = stds_vec[i]; // 当前描述符
    STDesc_LOC position;          // 描述符的哈希键
    int best_index = 0;
    STDesc_LOC best_position;
    // 设置匹配距离的阈值
    double dis_threshold =
        src_std.side_length_.norm() * config_setting_.rough_dis_threshold_; // 距离阈值
    // 遍历周围的体素来找到可能的匹配
    for (auto voxel_inc : voxel_round)
    {
      // 边长相近的哈希键
      position.x = (int)(src_std.side_length_[0] + voxel_inc[0]);
      position.y = (int)(src_std.side_length_[1] + voxel_inc[1]);
      position.z = (int)(src_std.side_length_[2] + voxel_inc[2]);
      Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                   (double)position.y + 0.5,
                                   (double)position.z + 0.5);
      // 判断当前描述符与体素的中心距离是否小于1.5，这是一个粗略的筛选，用于缩小搜索范围
      if ((src_std.side_length_ - voxel_center).norm() < 1.5)
      {

        auto iter = data_base_.find(position);
        if (iter != data_base_.end())
        {
          // 对于每一个匹配的描述符，检查它们之间的距离
          for (size_t j = 0; j < data_base_[position].size(); j++)
          {
            // 只有当描述符与数据库中的描述符之间的帧差大于一个阈值时，才进行匹配
            if ((src_std.frame_id_ - data_base_[position][j].frame_id_) >
                config_setting_.skip_near_num_)
            {
              // 超过阈值，则判断STDesc对象和匹配的STDesc对象的边长距离
              double dis =
                  (src_std.side_length_ - data_base_[position][j].side_length_)
                      .norm();
              // 如果距离小于阈值，进一步检查顶点附加信息的差异
              if (dis < dis_threshold)
              {
                // 距离小于阈值，则计算两个描述符顶点强度的差异
                dis_match_cnt++;
                double vertex_attach_diff =
                    2.0 *
                    (src_std.vertex_attached_ -
                     data_base_[position][j].vertex_attached_)
                        .norm() /
                    (src_std.vertex_attached_ +
                     data_base_[position][j].vertex_attached_)
                        .norm();
                // 如果顶点强度的差异小于预设阈值，标记这两个描述符为匹配
                if (vertex_attach_diff <
                    config_setting_.vertex_diff_threshold_)
                {
                  // 如果差异小于阈值，则标记为有用的匹配，匹配的位置添加到useful_match_position中
                  final_match_cnt++;
                  useful_match[i] = true;
                  useful_match_position[i].push_back(position);
                  useful_match_index[i].push_back(j);
                }
              }
            }
          }
        }
      }
    }
  }
  // std::cout << "dis match num:" << dis_match_cnt
  //           << ", final match num:" << final_match_cnt << std::endl;

  // 3. 记录匹配结果
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  for (size_t i = 0; i < useful_match.size(); i++)
  {
    if (useful_match[i])
    {
      for (size_t j = 0; j < useful_match_index[i].size(); j++)
      {
        match_array[data_base_[useful_match_position[i][j]]
                              [useful_match_index[i][j]]
                                  .frame_id_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        match_index_vec.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_id_);
      }
    }
  }

  // select candidate according to the matching score
  // 根据匹配得分选择候选者
  // 遍历从最高得分找到指定数量的候选者
  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++)
  {
    // 遍历找到最高得分
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < MAX_FRAME_N; i++)
    {
      if (match_array[i] > max_vote)
      {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    // 判断最高得分是否超过5
    STDMatchList match_triangle_list;
    if (max_vote_index >= 0 && max_vote >= 5)
    {
      // 得分超过5，则标记为候选者，
      match_array[max_vote_index] = 0;
      match_triangle_list.match_id_.first = current_frame_id_;
      match_triangle_list.match_id_.second = max_vote_index;
      for (size_t i = 0; i < index_recorder.size(); i++)
      {
        if (match_index_vec[i] == max_vote_index)
        {
          std::pair<STDesc, STDesc> single_match_pair;
          single_match_pair.first = stds_vec[index_recorder[i][0]];
          single_match_pair.second =
              data_base_[useful_match_position[index_recorder[i][0]]
                                              [index_recorder[i][1]]]
                        [useful_match_index[index_recorder[i][0]]
                                           [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
      candidate_matcher_vec.push_back(match_triangle_list);
    }
    else
    {
      break;
    }
  }
}

// Get the best candidate frame by geometry check
/**
 * @brief 通过几何验证得到最好的候选者,计算其得分，并确定相对姿势变换。
 *
 * @param candidate_matcher 输入：待验证的候选匹配
 * @param verify_score 输出：验证得分
 * @param relative_pose 输出：相对姿势变换
 * @param sucess_match_vec 输出：成功的匹配向量
 */
void STDescManager::candidate_verify(
    const STDMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<STDesc, STDesc>> &sucess_match_vec)
{
  sucess_match_vec.clear();
  // 计算需要跳过的匹配对长度和实际使用的匹配对数量
  int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
  int use_size = candidate_matcher.match_list_.size() / skip_len;
  // 设定距离阈值
  double dis_threshold = 3.0;
  // 初始化索引和投票列表
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  for (size_t i = 0; i < index.size(); i++)
  {
    index[i] = i;
  }
  std::mutex mylock;

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  // 对每一个匹配对进行验证
  for (size_t i = 0; i < use_size; i++)
  {
    auto single_pair = candidate_matcher.match_list_[i * skip_len];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    // 解决三角形问题以获取旋转和平移
    triangle_solver(single_pair, test_t, test_rot);
    // 遍历候选匹配列表，验证每一个匹配对
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++)
    {
      auto verify_pair = candidate_matcher.match_list_[j];
      // 对三个顶点进行变换并计算距离
      // A, B, C是三角形的三个顶点
      Eigen::Vector3d A = verify_pair.first.vertex_A_;
      Eigen::Vector3d A_transform = test_rot * A + test_t;
      Eigen::Vector3d B = verify_pair.first.vertex_B_;
      Eigen::Vector3d B_transform = test_rot * B + test_t;
      Eigen::Vector3d C = verify_pair.first.vertex_C_;
      Eigen::Vector3d C_transform = test_rot * C + test_t;
      double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
      double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
      double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
      // 如果所有顶点的变换后的距离都小于阈值，则为该匹配对投票
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold)
      {
        vote++;
      }
    }
    // 更新投票列表
    mylock.lock();
    vote_list[i] = vote;
    mylock.unlock();
  }
  // 查找得票最多的匹配对
  int max_vote_index = 0;
  int max_vote = 0;
  for (size_t i = 0; i < vote_list.size(); i++)
  {
    if (max_vote < vote_list[i])
    {
      max_vote_index = i;
      max_vote = vote_list[i];
    }
  }
  // 如果最大的投票数大于等于4，那么计算相对姿势并更新成功的匹配向量
  if (max_vote >= 4)
  {
    // 用得票最多的匹配对计算最佳的旋转和平移
    auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
    int vote = 0;
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;
    triangle_solver(best_pair, best_t, best_rot);
    relative_pose.first = best_t;
    relative_pose.second = best_rot;
    // 遍历候选匹配列表，验证每一个匹配对
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++)
    {
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.vertex_A_;
      Eigen::Vector3d A_transform = best_rot * A + best_t;
      Eigen::Vector3d B = verify_pair.first.vertex_B_;
      Eigen::Vector3d B_transform = best_rot * B + best_t;
      Eigen::Vector3d C = verify_pair.first.vertex_C_;
      Eigen::Vector3d C_transform = best_rot * C + best_t;
      double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
      double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
      double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold)
      {
        sucess_match_vec.push_back(verify_pair);
      }
    }
    verify_score = plane_geometric_verify(
        plane_cloud_vec_.back(),
        plane_cloud_vec_[candidate_matcher.match_id_.second], relative_pose);
  }
  else
  {
    verify_score = -1;
  }
}

/**
 * @brief 函数的目的是解决两个三角形之间的旋转和平移问题。它接收一个由两个STDesc构成的对，并输出一个旋转矩阵和一个平移向量。
 *
 * @param std_pair 两个STDesc构成的对
 * @param t 平移向量
 * @param rot 旋转矩阵
 */
void STDescManager::triangle_solver(std::pair<STDesc, STDesc> &std_pair,
                                    Eigen::Vector3d &t, Eigen::Matrix3d &rot)
{
  // 初始化两个3x3矩阵为零
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  // 为源三角形的每个顶点减去其中心，将结果存储在src矩阵中
  src.col(0) = std_pair.first.vertex_A_ - std_pair.first.center_;
  src.col(1) = std_pair.first.vertex_B_ - std_pair.first.center_;
  src.col(2) = std_pair.first.vertex_C_ - std_pair.first.center_;
  // 对参考三角形执行相同的操作，结果存储在ref矩阵中
  ref.col(0) = std_pair.second.vertex_A_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.vertex_B_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.vertex_C_ - std_pair.second.center_;
  // 计算协方差矩阵
  Eigen::Matrix3d covariance = src * ref.transpose();
  // 使用SVD (奇异值分解) 对协方差矩阵进行分解
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);
  // 获取V和U矩阵
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  // 计算旋转矩阵
  rot = V * U.transpose();
  // 如果旋转矩阵的行列式小于0，需要进行调整
  if (rot.determinant() < 0)
  {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  // 计算平移向量
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

/**
 * @brief 函数的目的是基于给定的刚性变换验证两个点云之间的几何平面匹配情况。
 * 它返回的是匹配成功的平面的比例。
 *
 * @param source_cloud
 * @param target_cloud
 * @param transform
 * @return double
 */
double STDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform)
{
  // 解构刚性变换为旋转矩阵和平移向量
  Eigen::Vector3d t = transform.first;
  Eigen::Matrix3d rot = transform.second;

  // 初始化KD树，用于高效地查找最近邻点
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  // 将目标点云的坐标转换为pcl::PointXYZ格式，并存储在input_cloud中
  for (size_t i = 0; i < target_cloud->size(); i++)
  {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  kd_tree->setInputCloud(input_cloud);

  // 初始化变量
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  // 从配置中读取法向量和距离阈值
  double normal_threshold = config_setting_.normal_threshold_;
  double dis_threshold = config_setting_.dis_threshold_;
  // 遍历源点云中的每一个点
  for (size_t i = 0; i < source_cloud->size(); i++)
  {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    // 将当前点及其法向量进行刚性变换
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    // 使用KD树在目标点云中查找K个最近邻点
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    int K = 3;
    if (kd_tree->nearestKSearch(use_search_point, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0)
    {
      // 遍历找到的K个最近邻点
      for (size_t j = 0; j < K; j++)
      {
        pcl::PointXYZINormal nearstPoint =
            target_cloud->points[pointIdxNKNSearch[j]];
        // 计算与最近邻点的法向量和点到平面的距离
        Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
        Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                            nearstPoint.normal_z);
        Eigen::Vector3d normal_inc = ni - tni;
        Eigen::Vector3d normal_add = ni + tni;
        double point_to_plane = fabs(tni.transpose() * (pi - tpi));
        // 如果法向量和点到平面的距离都满足阈值，则认为此点匹配成功
        if ((normal_inc.norm() < normal_threshold ||
             normal_add.norm() < normal_threshold) &&
            point_to_plane < dis_threshold)
        {
          useful_match++;
          break;
        }
      }
    }
  }
  return useful_match / source_cloud->size();
}

/**
 * @brief STDescManager类的方法，执行基于平面几何的迭代最近点（ICP）配准。
 *
 * @param source_cloud 输入：带有法线的源点云
 * @param target_cloud 输入：带有法线的目标点云
 * @param transform 输出：由平移矢量和旋转矩阵组成的变换
 */
void STDescManager::PlaneGeomrtricIcp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform)
{
  // 初始化KD树，用于高效的点云最近邻搜索
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  // 将目标点云转换为只有XYZ的点云，以供KD树使用
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++)
  {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  // 为KD树设置输入的点云
  kd_tree->setInputCloud(input_cloud);
  // 使用ceres库设置问题和参数块
  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  ceres::Problem problem;
  ceres::LossFunction *loss_function = nullptr;
  // 提取输入的变换，并转化为四元数和平移矢量的形式
  Eigen::Matrix3d rot = transform.second;
  Eigen::Quaterniond q(rot);
  Eigen::Vector3d t = transform.first;
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};

  // 添加四元数和平移矢量为问题的参数块
  problem.AddParameterBlock(para_q, 4, quaternion_manifold);
  problem.AddParameterBlock(para_t, 3);

  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
  // 用于KD树搜索的变量
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  int useful_match = 0;
  // 对于源点云中的每一个点，找到目标点云中的最近点
  for (size_t i = 0; i < source_cloud->size(); i++)
  {
    // 获取当前源点云中的点及其法线
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    // 将点的位置应用初始的旋转和平移变换
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    // 创建一个新的点用于KD树搜索
    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    // 应用初始旋转到法线
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;

    // 使用KD树搜索目标点云中离变换后的源点最近的点
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0)
    {
      // 获取最近点
      pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[0]];
      // 提取最近点的位置和法线
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);
      // 计算源点和目标点法线之间的增量和和差
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;

      // 计算点到点的距离和点到平面的距离
      double point_to_point_dis = (pi - tpi).norm();
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));

      // 根据法线和距离的阈值条件来检查这个匹配是否有用
      if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
           normal_add.norm() < config_setting_.normal_threshold_) &&
          point_to_plane < config_setting_.dis_threshold_ &&
          point_to_point_dis < 3)
      {
        // 如果满足条件，则认为这是一个有用的匹配
        useful_match++;
        // 定义ceres库的代价函数指针
        ceres::CostFunction *cost_function;
        // 提取当前源点的位置和法线
        Eigen::Vector3d curr_point(source_cloud->points[i].x,
                                   source_cloud->points[i].y,
                                   source_cloud->points[i].z);
        Eigen::Vector3d curr_normal(source_cloud->points[i].normal_x,
                                    source_cloud->points[i].normal_y,
                                    source_cloud->points[i].normal_z);
        // 使用PlaneSolver类创建一个新的代价函数，该函数将基于当前点、其法线、
        // 最近的目标点及其法线来计算代价
        cost_function = PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
      }
    }
  }
  // ceres求解器的设置
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  // 求解优化问题
  ceres::Solve(options, &problem, &summary);
  // 从求解结果中获取优化后的变换
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);
  transform.first = t;
  transform.second = rot;
  // std::cout << "useful match for icp:" << useful_match << std::endl;
}

/**
 * @brief OctoTree类的方法，用于初始化平面
 *
 */
void OctoTree::init_plane()
{
  // 初始化变量
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  // 计算点云的协方差矩阵和中心点
  for (auto pi : voxel_points_)
  {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();

  // 对协方差矩阵进行特征值分解
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  // 提取实数部分的特征值
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  // 检查最小特征值是否小于配置中的阈值
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_)
  {
    // 如果是，则认为点云数据分布在一个平面上，并计算该平面的属性
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    // 计算平面的截距和其他属性
    plane_ptr_->intercept_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                               plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                               plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  }
  else
  {
    // 否则，此体素的点不分布在一个平面上
    plane_ptr_->is_plane_ = false;
  }
}

/**
 * @brief  初始化OctoTree
 *
 */
void OctoTree::init_octo_tree()
{
  // 如果体素中的点数超过了配置中的初始数量，则尝试初始化平面
  if (voxel_points_.size() > config_setting_.voxel_init_num_)
  {
    init_plane();
  }
}