#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class PointCloudAggregator {
public:
  PointCloudAggregator() : nh_(), nhp_("~"), count_(0) {
    nhp_.param("batch_size", batch_size_, 20);
    sub_ = nh_.subscribe("input_point_cloud", 10, &PointCloudAggregator::pointCloudCallback, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>("output_point_cloud", 10);
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle nhp_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  pcl::PointCloud<pcl::PointXYZI> accumulated_cloud_;
  int count_;
  int batch_size_;

  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(*msg, cloud);
    accumulated_cloud_ += cloud;
    count_++;

    if (count_ >= batch_size_) {
      sensor_msgs::PointCloud2 output_msg;
      pcl::toROSMsg(accumulated_cloud_, output_msg);
      output_msg.header = msg->header;
      pub_.publish(output_msg);

      accumulated_cloud_.clear();
      count_ = 0;
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "point_cloud_aggregator");
  PointCloudAggregator aggregator;
  ros::spin();
  return 0;
}
