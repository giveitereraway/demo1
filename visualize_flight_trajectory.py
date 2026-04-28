"""
飞机三维飞行轨迹可视化程序
读取飞行轨迹数据文件并绘制三维轨迹图
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, List, Tuple


def parse_flight_data(file_path: str) -> Dict[str, Dict[str, List]]:
    """
    解析飞行轨迹数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        包含各飞机轨迹数据的字典，格式为：
        {
            'A0100': {
                'time': [时间列表],
                'longitude': [经度列表],
                'latitude': [纬度列表],
                'altitude': [高度列表],
                'roll': [翻滚角列表],
                'pitch': [俯仰角列表],
                'yaw': [偏航角列表],
                'name': 飞机型号,
                'color': 颜色
            },
            ...
        }
    """
    flight_data = {}
    current_time = 0.0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 解析时间戳
            if line.startswith('#'):
                current_time = float(line[1:])
                continue
            
            # 解析飞机数据
            # 格式: A0100,T=lon|lat|alt|roll|pitch|yaw,Name=F16,Color=Red
            parts = line.split(',')
            if len(parts) < 3:
                continue
                
            aircraft_id = parts[0]
            
            # 解析位置和姿态数据
            if not parts[1].startswith('T='):
                continue
            data_str = parts[1][2:]  # 去掉 'T='
            data_values = [float(x) for x in data_str.split('|')]
            
            if len(data_values) < 6:
                continue
            
            longitude, latitude, altitude, roll, pitch, yaw = data_values
            
            # 解析名称和颜色
            name = ''
            color = ''
            for part in parts[2:]:
                if part.startswith('Name='):
                    name = part[5:]
                elif part.startswith('Color='):
                    color = part[6:]
            
            # 初始化飞机数据结构
            if aircraft_id not in flight_data:
                flight_data[aircraft_id] = {
                    'time': [],
                    'longitude': [],
                    'latitude': [],
                    'altitude': [],
                    'roll': [],
                    'pitch': [],
                    'yaw': [],
                    'name': name,
                    'color': color.lower()
                }
            
            # 添加数据点
            flight_data[aircraft_id]['time'].append(current_time)
            flight_data[aircraft_id]['longitude'].append(longitude)
            flight_data[aircraft_id]['latitude'].append(latitude)
            flight_data[aircraft_id]['altitude'].append(altitude)
            flight_data[aircraft_id]['roll'].append(roll)
            flight_data[aircraft_id]['pitch'].append(pitch)
            flight_data[aircraft_id]['yaw'].append(yaw)
    
    return flight_data


def plot_3d_trajectory(flight_data: Dict[str, Dict[str, List]], output_file: str = None):
    """
    绘制三维飞行轨迹图
    
    Args:
        flight_data: 解析后的飞行数据
        output_file: 输出图片文件路径（可选），如果不指定则显示图形
    """
    # 创建3D图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每架飞机绘制轨迹
    for aircraft_id, data in flight_data.items():
        longitude = np.array(data['longitude'])
        latitude = np.array(data['latitude'])
        altitude = np.array(data['altitude'])
        color = data['color']
        name = data['name']
        
        # 绘制轨迹线
        ax.plot(longitude, latitude, altitude, 
                color=color, 
                linewidth=2, 
                label=f"{aircraft_id} ({name}) - {color.capitalize()}",
                alpha=0.7)
        
        # 绘制起点（大圆点）
        ax.scatter(longitude[0], latitude[0], altitude[0], 
                  color=color, 
                  s=100, 
                  marker='o', 
                  edgecolors='black',
                  linewidths=2,
                  label=f"{aircraft_id} 起点")
        
        # 绘制终点（五角星）
        ax.scatter(longitude[-1], latitude[-1], altitude[-1], 
                  color=color, 
                  s=200, 
                  marker='*', 
                  edgecolors='black',
                  linewidths=2,
                  label=f"{aircraft_id} 终点")
        
        # 沿轨迹绘制质点标记（每隔一定间隔）
        step = max(1, len(longitude) // 20)  # 最多显示20个标记点
        ax.scatter(longitude[::step], latitude[::step], altitude[::step], 
                  color=color, 
                  s=30, 
                  marker='o',
                  alpha=0.5)
    
    # 设置坐标轴标签
    ax.set_xlabel('经度 (Longitude)', fontsize=12, labelpad=10)
    ax.set_ylabel('纬度 (Latitude)', fontsize=12, labelpad=10)
    ax.set_zlabel('海拔高度 (Altitude, m)', fontsize=12, labelpad=10)
    
    # 设置标题
    ax.set_title('飞机三维飞行轨迹可视化', fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图形已保存到: {output_file}")
    else:
        plt.show()


def print_flight_summary(flight_data: Dict[str, Dict[str, List]]):
    """
    打印飞行数据摘要信息
    
    Args:
        flight_data: 解析后的飞行数据
    """
    print("=" * 60)
    print("飞行轨迹数据摘要")
    print("=" * 60)
    
    for aircraft_id, data in flight_data.items():
        print(f"\n飞机ID: {aircraft_id}")
        print(f"  型号: {data['name']}")
        print(f"  颜色: {data['color'].capitalize()}")
        print(f"  数据点数量: {len(data['time'])}")
        print(f"  飞行时长: {data['time'][0]:.2f}s - {data['time'][-1]:.2f}s")
        print(f"  经度范围: {min(data['longitude']):.6f} - {max(data['longitude']):.6f}")
        print(f"  纬度范围: {min(data['latitude']):.6f} - {max(data['latitude']):.6f}")
        print(f"  高度范围: {min(data['altitude']):.2f}m - {max(data['altitude']):.2f}m")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    # 数据文件路径
    data_file = 'scripts/agent_follow_human'
    
    print("开始解析飞行轨迹数据...")
    
    # 解析数据
    flight_data = parse_flight_data(data_file)
    
    if not flight_data:
        print("错误：未能解析到有效的飞行数据！")
        return
    
    print(f"成功解析 {len(flight_data)} 架飞机的轨迹数据")
    
    # 打印数据摘要
    print_flight_summary(flight_data)
    
    # 绘制三维轨迹图
    print("\n正在生成三维轨迹可视化...")
    plot_3d_trajectory(flight_data)
    
    # 可选：保存图片
    # plot_3d_trajectory(flight_data, output_file='flight_trajectory_3d.png')


if __name__ == '__main__':
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
