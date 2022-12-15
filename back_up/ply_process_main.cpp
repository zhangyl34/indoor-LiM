#include <pcl/io/ply_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <vtkAutoInit.h>
#include <vtkRenderWindow.h>


/* noisy point cloud filter.*/
void plyFilter(const std::string file) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);           // 待滤波点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);  // 滤波后点云

	// 读入点云数据
	std::cout << "->正在读入点云..." << std::endl;
    std::cout << file << std::endl;
	pcl::PLYReader reader;
	reader.read(file, *cloud);
	std::cout << "\t\t<读入点云信息>\n" << *cloud << std::endl;

	// 统计滤波
	std::cout << "->正在进行统计滤波..." << std::endl;
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;  //创建滤波器对象
	sor.setInputCloud(cloud);							//设置待滤波点云
	sor.setMeanK(50);									//设置查询点近邻点的个数
	sor.setStddevMulThresh(1.0);						//设置标准差乘数，来计算是否为离群点的阈值
	//sor.setNegative(true);							//默认false，保存内点；true，保存滤掉的离群点
	sor.filter(*cloud_filtered);						//执行滤波，保存滤波结果于cloud_filtered

	//保存下采样点云
	std::cout << "->正在保存滤波点云..." << std::endl;
	pcl::PLYWriter writer;
	writer.write("StatisticalOutlierRemoval.ply", *cloud_filtered, true);
	std::cout << "\t\t<保存点云信息>\n" << *cloud_filtered << std::endl;

	//================================= 滤波前后对比可视化 ================================= ↓

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("滤波前后对比"));

	/*-----视口1-----*/
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); //设置第一个视口在X轴、Y轴的最小值、最大值，取值在0-1之间
	viewer->setBackgroundColor(0, 0, 0, v1); //设置背景颜色，0-1，默认黑色（0，0，0）
	viewer->addText("befor_filtered", 10, 10, "v1_text", v1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "befor_filtered_cloud", v1);

	/*-----视口2-----*/
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("after_filtered", 10, 10, "v2_text", v2);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "after_filtered_cloud", v2);

	/*-----设置相关属性-----*/
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "befor_filtered_cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "befor_filtered_cloud", v1);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "after_filtered_cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "after_filtered_cloud", v2);

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	//================================= 滤波前后对比可视化 ================================= ↑
}

/* from ply file to voxel map.*/
void plyMap(const std::string file, const float leaf = 0.1) {

	//point clouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>());

    // 读入点云数据
	std::cout << "->正在读入点云..." << std::endl;
    std::cout << file << std::endl;
	pcl::PLYReader reader;
	reader.read(file, *cloud);
	std::cout << "\t\t<读入点云信息>\n" << *cloud << std::endl;

    // 计算 AABB
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);
	std::cout << "AABB 信息：\n" << std::endl;
	std::cout << min_point_AABB.x << "~" << max_point_AABB.x << std::endl;
	std::cout << min_point_AABB.y << "~" << max_point_AABB.y << std::endl;
	std::cout << min_point_AABB.z << "~" << max_point_AABB.z << std::endl;

    // VoxelGrid filtering
	pcl::VoxelGrid<pcl::PointXYZ> fil;
	fil.setInputCloud(cloud);
	fil.setLeafSize(leaf, leaf, leaf);
	fil.filter(*cloud_filter);
	int filter_num = cloud_filter->points.size();
	std::cout << "降采样后的点云个数：" << filter_num << std::endl;

    // 保存数据为 txt 文件
	int dim_x = ceil((max_point_AABB.x - min_point_AABB.x)/leaf);
	int dim_y = ceil((max_point_AABB.y - min_point_AABB.y)/leaf);
	int dim_z = ceil((max_point_AABB.z - min_point_AABB.z)/leaf);
    Eigen::Tensor<bool, 3> occupy(dim_x,dim_y,dim_z);
    occupy.setZero();
	for (int i = 0; i < filter_num; i++) {
		int x = floor((cloud_filter->points[i].x - min_point_AABB.x) / leaf);
		int y = floor((cloud_filter->points[i].y - min_point_AABB.y) / leaf);
		int z = floor((cloud_filter->points[i].z - min_point_AABB.z) / leaf);
		occupy(x,y,z)=true;
	}
	std::ofstream OutFile("occupy.txt");
	OutFile << "# " << std::to_string(dim_x) << ' ' << std::to_string(dim_y) << ' ' 
	    << std::to_string(dim_z) << '\n';
	for (int i = 0; i < dim_x; i++) {
		for (int j = 0; j < dim_y; j++) {
			for (int k = 0; k < (dim_z-1); k++) {
			    OutFile << occupy(i, j, k) << ',';
			}
			OutFile << occupy(i, j, dim_z-1) << std::endl;
		}
	}
	OutFile.close();  // 关闭 Test.txt 文件

    // 渲染
    // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Voxelization"));
	// viewer->setBackgroundColor(0,0,0);
	// viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y,
	//     min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 1.0, "AABB");
	// viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
	//     pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "AABB");
    // for(int i = 0; i < filter_num; i++) {
	// 	double x = cloud_filter->points[i].x;
	// 	double y = cloud_filter->points[i].y;
	// 	double z = cloud_filter->points[i].z;
	// 	Eigen::Vector3f center(floor(x / leaf)*leaf + leaf/2, floor(y / leaf)*leaf + leaf/2, floor(z / leaf)*leaf + leaf/2);
	// 	Eigen::Quaternionf rotation(1,0,0,0);
	// 	std::string cube = "AABB"+std::to_string(i);
	// 	viewer->addCube(center, rotation, leaf, leaf, leaf, cube);
	// }
	// std::cout << "point cloud loaded!" << std::endl;

	// while (!viewer->wasStopped())
	// {
	// 	viewer->spinOnce(100);
	// 	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	// }
}

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

VTK_MODULE_INIT(vtkRenderingOpenGL2)
VTK_MODULE_INIT(vtkInteractionStyle)

inline double uniform_deviate(int seed)
{
	double ran = seed * (1.0 / (RAND_MAX + 1.0));
	return ran;
}

inline void randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3, Eigen::Vector4f& p)
{
	float r1 = static_cast<float> (uniform_deviate(rand()));
	float r2 = static_cast<float> (uniform_deviate(rand()));
	float r1sqr = std::sqrt(r1);
	float OneMinR1Sqr = (1 - r1sqr);
	float OneMinR2 = (1 - r2);
	a1 *= OneMinR1Sqr;
	a2 *= OneMinR1Sqr;
	a3 *= OneMinR1Sqr;
	b1 *= OneMinR2;
	b2 *= OneMinR2;
	b3 *= OneMinR2;
	c1 = r1sqr * (r2 * c1 + b1) + a1;
	c2 = r1sqr * (r2 * c2 + b2) + a2;
	c3 = r1sqr * (r2 * c3 + b3) + a3;
	p[0] = c1;
	p[1] = c2;
	p[2] = c3;
	p[3] = 0;
}

inline void randPSurface(vtkPolyData * polydata, std::vector<double> * cumulativeAreas, double totalArea, Eigen::Vector4f& p, bool calcNormal, Eigen::Vector3f& n)
{
	float r = static_cast<float> (uniform_deviate(rand()) * totalArea);

	std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
	vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

	double A[3], B[3], C[3];
	vtkIdType npts = 0;
	vtkIdType *ptIds = NULL;
	polydata->GetCellPoints(el, npts, ptIds);
	polydata->GetPoint(ptIds[0], A);
	polydata->GetPoint(ptIds[1], B);
	polydata->GetPoint(ptIds[2], C);
	if (calcNormal)
	{
		// OBJ: Vertices are stored in a counter-clockwise order by default
		Eigen::Vector3f v1 = Eigen::Vector3f(A[0], A[1], A[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
		Eigen::Vector3f v2 = Eigen::Vector3f(B[0], B[1], B[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
		n = v1.cross(v2);
		n.normalize();
	}
	randomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
		float(B[0]), float(B[1]), float(B[2]),
		float(C[0]), float(C[1]), float(C[2]), p);
}

void uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal, pcl::PointCloud<pcl::PointNormal> & cloud_out)
{
	polydata->BuildCells();
	vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

	double p1[3], p2[3], p3[3], totalArea = 0;
	std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
	size_t i = 0;
	vtkIdType npts = 0, *ptIds = NULL;
	for (cells->InitTraversal(); cells->GetNextCell(npts, ptIds); i++)
	{
		polydata->GetPoint(ptIds[0], p1);
		polydata->GetPoint(ptIds[1], p2);
		polydata->GetPoint(ptIds[2], p3);
		totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
		cumulativeAreas[i] = totalArea;
	}

	cloud_out.points.resize(n_samples);
	cloud_out.width = static_cast<pcl::uint32_t> (n_samples);
	cloud_out.height = 1;

	for (i = 0; i < n_samples; i++)
	{
		Eigen::Vector4f p;
		Eigen::Vector3f n;
		randPSurface(polydata, &cumulativeAreas, totalArea, p, calc_normal, n);
		cloud_out.points[i].x = p[0];
		cloud_out.points[i].y = p[1];
		cloud_out.points[i].z = p[2];
		if (calc_normal)
		{
			cloud_out.points[i].normal_x = n[0];
			cloud_out.points[i].normal_y = n[1];
			cloud_out.points[i].normal_z = n[2];
		}
	}
}

/* from obj file to ply file.*/
void obj2ply(std::string file, const int SAMPLE_POINTS_ = 10000, const float leaf_size = 0.002f, const bool write_normals = true) {

    // read obj file.
	vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkOBJReader> readerQuery = vtkSmartPointer<vtkOBJReader>::New();
	readerQuery->SetFileName(file.c_str());
	readerQuery->Update();
	polydata1 = readerQuery->GetOutput();

	//make sure that the polygons are triangles!
	vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
	std::cout << "VTK_MAJOR_VERSION: " << VTK_MAJOR_VERSION << std::endl;
#if VTK_MAJOR_VERSION < 6
	triangleFilter->SetInput(polydata1);
#else
	triangleFilter->SetInputData(polydata1);
#endif
	triangleFilter->Update();
	vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
	triangleMapper->Update();
	polydata1 = triangleMapper->GetInput();

    // uniform sampling
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_1(new pcl::PointCloud<pcl::PointNormal>);
	uniform_sampling(polydata1, SAMPLE_POINTS_, write_normals, *cloud_1);

	// voxel grid filter
	VoxelGrid<PointNormal> grid_;
	grid_.setInputCloud(cloud_1);
	grid_.setLeafSize(leaf_size, leaf_size, leaf_size);
	pcl::PointCloud<pcl::PointNormal>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointNormal>);
	grid_.filter(*voxel_cloud);

    // visualize
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("cart model"));
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 1.0, 1.0, v1); //设置第一个视口在X轴、Y轴的最小值、最大值，取值在0-1之间
	viewer->setBackgroundColor(0, 0, 0, v1); //设置背景颜色，0-1，默认黑色（0，0，0）
	viewer->addText("befor_filtered", 10, 10, "v1_text", v1);

    if (write_normals) {
		// write normal vector.
	    savePCDFileBinary("stl_normal.pcd", *voxel_cloud);

		viewer->addPointCloud<pcl::PointNormal>(voxel_cloud, "befor_filtered_cloud", v1);
	}
	else {
		// write xyz, without normal vector.
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*voxel_cloud, *cloud_xyz);
		pcl::PLYWriter writer;
		writer.write("stl_xyz.ply", *cloud_xyz, true);

		viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, "befor_filtered_cloud", v1);
	}

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "befor_filtered_cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "befor_filtered_cloud", v1);
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	
}


/* input1: file name; input2: voxel leaf size.*/
int main(int argc,char *argv[])
{
	//plyFilter(argv[1]);

    // if (argc == 2) {
	// 	plyMap(argv[1]);
	// }
	// else if (argc == 3) {
    //     plyMap(argv[1], std::stof(argv[2]));
	// }
	
    if (argc == 2) {
		obj2ply(argv[1]);
	}
	else if (argc == 4) {
        obj2ply(argv[1], std::stoi(argv[2]), std::stof(argv[3]));
	}
	


	return 0;
}
