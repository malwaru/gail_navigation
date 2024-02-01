#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "std_msgs/msg/string.hpp"
#include "rosgraph_msgs/msg/clock.hpp"


// #include "turtlesim/msg/pose.hpp"

using namespace std::chrono_literals;

class PlaybackNode : public rclcpp::Node
{
  public:
    PlaybackNode(const std::string & bag_filename)
    : Node("ros_playback_node")
    {
      // publisher_ = this->create_publisher<turtlesim::msg::Pose>("/turtle1/pose", 10);
      clicked_point_pub_ =  this->create_publisher<geometry_msgs::msg::PointStamped>("/clicked_point", 10);
      clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
      cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
      diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
      events_read_split_pub_ = create_publisher<std_msgs::msg::String>("/events/read_split", 10);
      framos_camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/framos/camera_info", 10);
      framos_depth_camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/framos/depth/camera_info", 10);
      framos_depth_image_raw_pub_ = create_publisher<sensor_msgs::msg::Image>("/framos/depth/image_raw", 10);
      framos_image_raw_pub_ = create_publisher<sensor_msgs::msg::Image>("/framos/image_raw", 10);
      goal_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);
      gps_fix_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>("/gps/fix", 10);
      imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/imu", 10);
      initialpose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 10);
      joint_states_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
      odometry_filtered_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", 10);
      odometry_wheel_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/wheel", 10);
      parameter_events_pub_ = create_publisher<rcl_interfaces::msg::ParameterEvent>("/parameter_events", 10);
      set_pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/set_pose", 10);
      tf_pub_ = create_publisher<geometry_msgs::msg::TransformStamped>("/tf", 10);
      tf_static_pub_ = create_publisher<geometry_msgs::msg::TransformStamped>("/tf_static", 10);

      // Create a timer that will fire every 100ms
      timer_ = this->create_wall_timer(
          100ms, std::bind(&PlaybackNode::timer_callback, this));

      // Open the bag file
      reader_.open(bag_filename);
    }

  private:
    void timer_callback()
    {
      while (reader_.has_next()) {

        rosbag2_storage::SerializedBagMessageSharedPtr msg   = reader_.read_next();   
        if (msg->topic_name == "/clicked_point") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::PointStamped::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::PointStamped>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          clicked_point_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/clock") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          rosgraph_msgs::msg::Clock::SharedPtr ros_msg = std::make_shared<rosgraph_msgs::msg::Clock>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          clock_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/cmd_vel") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::Twist::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::Twist>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          cmd_vel_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/diagnostics") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          diagnostic_msgs::msg::DiagnosticArray::SharedPtr ros_msg = std::make_shared<diagnostic_msgs::msg::DiagnosticArray>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          diagnostics_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/events/read_split") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          std_msgs::msg::String::SharedPtr ros_msg = std::make_shared<std_msgs::msg::String>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          events_read_split_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/framos/camera_info") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::CameraInfo::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          framos_camera_info_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/framos/depth/camera_info") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::CameraInfo::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          framos_depth_camera_info_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/framos depth/image_raw") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::Image::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::Image>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          framos_depth_image_raw_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/framos/image_raw") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::Image::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::Image>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          framos_image_raw_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/goal_pose") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::PoseStamped::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::PoseStamped>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          goal_pose_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/gps/fix") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::NavSatFix::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::NavSatFix>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          gps_fix_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/imu") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::Imu::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::Imu>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          imu_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/initialpose") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::PoseWithCovarianceStamped>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          initialpose_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/joint_states") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data); 
          sensor_msgs::msg::JointState::SharedPtr ros_msg = std::make_shared<sensor_msgs::msg::JointState>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          joint_states_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/odometry/filtered") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          nav_msgs::msg::Odometry::SharedPtr ros_msg = std::make_shared<nav_msgs::msg::Odometry>();
          serialization_odometry_filtered.deserialize_message(&serialized_msg, ros_msg.get());
          odometry_filtered_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/odometry/wheel") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          nav_msgs::msg::Odometry::SharedPtr ros_msg = std::make_shared<nav_msgs::msg::Odometry>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          odometry_wheel_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/parameter_events") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          rcl_interfaces::msg::ParameterEvent::SharedPtr ros_msg = std::make_shared<rcl_interfaces::msg::ParameterEvent>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          parameter_events_pub_->publish(*ros_msg);
        }

        else if (msg->topic_name == "/set_pose") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::PoseWithCovarianceStamped>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          set_pose_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name == "/tf") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::TransformStamped::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::TransformStamped>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          tf_pub_->publish(*ros_msg);
        }
        else if (msg->topic_name ==  "/tf_static") {
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          geometry_msgs::msg::TransformStamped::SharedPtr ros_msg = std::make_shared<geometry_msgs::msg::TransformStamped>();
          serialization_.deserialize_message(&serialized_msg, ros_msg.get());
          tf_static_pub_->publish(*ros_msg);
        }
        else {
          std::cout << "Unknown topic: " << msg->topic_name << std::endl;
        }
        break;
      }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    // rclcpp::Publisher<turtlesim::msg::Pose>::SharedPtr publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr clicked_point_pub_;
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr events_read_split_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr framos_camera_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr framos_depth_camera_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr framos_depth_image_raw_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr framos_image_raw_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr gps_fix_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_filtered_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_wheel_pub_;
    rclcpp::Publisher<rcl_interfaces::msg::ParameterEvent>::SharedPtr parameter_events_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr set_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr tf_static_pub_;
    

    rclcpp::Serialization<nav_msgs::msg::Odometry> serialization_odometry_filtered;
    rclcpp::Serialization<sensor_msgs::msg::NavSatFix> serialization_nav_sat_fix;
    rclcpp::Serialization<sensor_msgs::msg::Imu> serialization_imu;
    rclcpp::Serialization<geometry_msgs::msg::PointStamped> serialization_point_stamped;
    rclcpp::Serialization<geometry_msgs::msg::PoseStamped> serialization_pose_stamped;
    rclcpp::Serialization<geometry_msgs::msg::PoseWithCovarianceStamped> serialization_pose_with_covariance_stamped;
    rclcpp::Serialization<geometry_msgs::msg::TransformStamped> serialization_transform_stamped;
    rclcpp::Serialization<rosgraph_msgs::msg::Clock> serialization_clock;
    rclcpp::Serialization<sensor_msgs::msg::CameraInfo> serialization_camera_info;
    rclcpp::Serialization<sensor_msgs::msg::Image> serialization_image;
    rclcpp::Serialization<sensor_msgs::msg::Image> serialization_;



    rosbag2_cpp::Reader reader_;
};

int main(int argc, char ** argv)
{
  // if (argc != 2) {
  //   std::cerr << "Usage: " << argv[0] << " <bag>" << std::endl;
  //   return 1;
  // }
  std::string bag_filename = "/home/malika/Documents/workspace/Bonn_Stuff/master_thesis/code/data/trajectories/traj1/rosbag2_2023_12_10-18_14_18_0.db3";

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PlaybackNode>(bag_filename));
  rclcpp::shutdown();

  return 0;
}