import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import xlrd  # 读取excel地铁站点与站名对应信息
import random
import linecache
from scipy.interpolate import interpolate
import numpy as np
import openpyxl

# 读取excel文件内容
def read_xlrd(excel_File):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)
    for rowNum in range(1, table.nrows):
        rowVale = table.row_values(rowNum)
        for colNum in range(table.ncols - 1):
            if colNum == 1:
                num = rowVale[colNum]
            if colNum == 3:
                station = rowVale[colNum]
        stations[num] = station

# 保存信息函数
def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print(filename + "文件保存成功!")

# 删除信息函数
def delete_info(file_name):
    file = open(file_name, 'r+')
    file.truncate()

#  将数据写入新文件，写入column列
def data_write(file_path, data, sheet_name, start_row, start_column):
    wb = openpyxl.load_workbook(file_path)
    ws = wb[sheet_name]
    for element in data:
        c = ws.cell(column=start_column, row=start_row)
        c.value = element
        start_row += 1
    wb.save(file_path)

# 结构检测函数，检测网络结构函数，网络大小，联通性，最大子图，调用函数绘制网络
def test_network(graph, filename):
    file = open(filename, 'a')
    file.write("网络结构图的节点数：" + str(graph.number_of_nodes()) + '\n'
                                                             "网络结构图的边数目：" + str(graph.number_of_edges()) + '\n'
                                                                                                           "当前网络最大联通子图节点数：" + str(
        len(max(nx.connected_components(graph), key=len))))
    file.close()
    nx.draw_networkx(graph)
    plt.show()

# 求站点度ki/k_max
def get_per_degree(i):
    # 平均度的列表
    aver_degree = []
    for ki in node_degree:
        aver_degree.append(ki / k_max)
    return aver_degree[i - 1]

# 求Bi/B_max
def get_per_betweenness(i):
    # 存放各站点介数中心性的字典
    score = nx.betweenness_centrality(G)
    # 求出最大的介数中心性
    B_max = max(score.values())
    # 求Bi/Bmax
    aver_shortest_path = []
    for x in score.values():
        aver_shortest_path.append(x / B_max)
    return aver_shortest_path[i - 1]

# 站点容量的初始化,节点i,参数是容忍系数a和权重系数w
def init_station_capacity(a, w):
    capacity = []
    for i in nodes:
        q = get_per_degree(i)
        b = get_per_betweenness(i)
        capacity.append((1 + a) * (w * q + (1 - w) * b))
    return capacity

# 初始化站点负载模拟
def init_station_load(a, w):
    aver_station_load = []
    for i in nodes:
        q = get_per_degree(i)
        b = get_per_betweenness(i)
        aver_station_load.append((w * q + (1 - w) * b))
    return aver_station_load

# 模拟失效节点函数
def random_attack(graph, degree):
    msg = input("请输入失效模型：0-停止模拟，1-随机失效，2-最大度失效，3-最大介数站点失效，4-新街口站点失效\n")
    if msg == '1':
        return random.randint(1, graph.number_of_nodes())
    elif msg == '2':
        return int(degree.index(max(degree)) + 1)
    elif msg == '3':
        score = nx.betweenness_centrality(graph)
        b_max = max(score.values())
        return list(score.keys())[list(score.values()).index(b_max)]
    elif msg == '4':
        return int(8)
    elif msg == '0':
        return 0

# 归一化处理函数
def normalization(list_argument):
    temp_list = []
    for list_element in list_argument:
        temp_list.append(list_element / stations_nums)
    return temp_list


def load_restribution(graph, load, failure_node_number, failed_list):
    # 更新load字典的值
    current_load = load[failure_node_number]
    print("当前失效节点负载为:" + str(current_load))
    # 存邻居节点的列表
    neighbors_nodes = [*graph.neighbors(failure_node_number)]
    neighbors_nodes = set(neighbors_nodes).difference(set(failed_list))
    print("邻居节点为: " + str(neighbors_nodes))
    # 求邻居节点的当前负载总和sum_neighbor_load
    sum_neighbor_load = 0.0
    for neighbors_node in neighbors_nodes:
        # 所有邻居节点负载总和sum_neighbor_load,也就是式子中的分母
        sum_neighbor_load += float(load[neighbors_node])
    # print("邻居节点总负载为: " + str(sum_neighbor_load))
    # 更新后的节点负载字典
    for neighbors_node in neighbors_nodes:
        load[neighbors_node] = load[neighbors_node] * (1 + float(current_load) / sum_neighbor_load)
    # print("更新后邻居节点负载为：")
    # for neighbors_node in neighbors_nodes:
    #     print(str(load[neighbors_node]) + " ")
    # 标记失效节点
    load[failure_node_number] = 'f'
    return load


def get_graph_info(graph, components_list, live_list):
    try:
        largest_components = max(nx.connected_components(G))
    except ValueError:
        components_list.append(0)
        print("当前图中最大联通子图节点数: " + str(0))
    else:
        largest_components = max(nx.connected_components(G))
        components_list.append(len(largest_components))
        print("当前图中最大联通子图节点数: " + str(len(largest_components)))
    live_num = graph.number_of_nodes()
    print("当前图中未失效节点数为: " + str(live_num))
    live_list.append(live_num)

# 级联失效主函数
def cascading_failure_node(graph, capacity_file, load_file, failure_node_number):
    # 存入load和capacity字典
    load = {}
    capacity = {}
    for i in range(1, stations_nums + 1):
        load[i] = float((linecache.getline(load_file, i).strip()))
        capacity[i] = float((linecache.getline(capacity_file, i).strip()))
    print("当前失效节点为：" + str(failure_node_number), end='\t')
    # 当前失效节点的负载，待分配负载
    load = load_restribution(graph, load, failure_node_number, [failure_node_number])
    # 删除失效节点
    graph.remove_node(failure_node_number)
    print("节点:" + str(nodes[failure_node_number - 1]) + "失效！信息已删除！")
    get_graph_info(graph, largest_components_list, not_fail_nodes)

    for ki in range(2, 51):
        print("---------------第" + str(ki) + "级失效！--------------")
        # 检查load字典
        failed_lists = []
        neighbors_judge = {}
        for key, values in load.items():
            if values != 'f':
                neighbors_judge[key] = float(capacity[key]) - float(load[key])
                if neighbors_judge[key] < 0:
                    print("第" + str(key) + "号节点失效！")
                    failed_lists.append(key)
        for failed_node in failed_lists:
            load = load_restribution(graph, load, failed_node, failed_lists)
        for failed_node in failed_lists:
            graph.remove_node(failed_node)
            print("节点:" + str(nodes[failed_node - 1]) + "失效！信息已删除！")
        get_graph_info(graph, largest_components_list, not_fail_nodes)


if __name__ == '__main__':
    stations = {}  # 用来存放序号和站名的字典
    # 读取地铁网络信息
    excelFile = 'D:/UniCourse/Srt/code/Network-cascade/data/allsubway.xlsx'
    read_xlrd(excel_File=excelFile)
    # 建图G
    G = nx.Graph()
    stations_nums = len(stations.keys())  # 图中站点总数
    nodes = range(1, len(stations.keys()) + 1)  # 图中总节点数
    # 导入159个地铁站点
    G.add_nodes_from(nodes)
    edges = pd.read_csv('网络拓扑数据.txt', sep=',', header=None)
    edge_lists = [tuple(xi) for xi in edges.values]
    for edge_list in edge_lists:
        G.add_edge(*edge_list)
    # 存放各站点度的列表
    node_degree = []
    for number in stations.keys():
        node_degree.append(G.degree(number))
    # 网络最大度
    k_max = max(node_degree)

    # 构建仿真模型
    argument_a = input("请输入站点容量的容忍系数a:")
    argument_w = input("请输入权重系数w,范围[0,1]:")
    print("正在生成信息并保存，请等待！")
    text_save('capacity.txt', init_station_capacity(float(argument_a), float(argument_w)))
    text_save('load.txt', init_station_load(float(argument_a), float(argument_w)))
    test_network(G, 'G.txt')

    first_fail_node = random_attack(G, node_degree)
    largest_components_list = [stations_nums]
    not_fail_nodes = [stations_nums]
    cascading_failure_node(G, 'capacity.txt', 'load.txt', first_fail_node)

    print("当前剩余节点: ")
    print(G.nodes())
    print("未失效节点数：")
    print(not_fail_nodes)
    print("最大联通子图节点数：")
    print(largest_components_list)

    with open('data.txt', 'a+') as f:
        f.writelines('\n' + "a=" + argument_a + ",w=" + argument_w + '\n')
        f.writelines("最大联通子图节点数 " + str(largest_components_list) + '\n')
        f.writelines("未失效节点数 " + str(not_fail_nodes) + '\n')

    # 设置显示中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    x_new = np.arange(0, 51)
    # 插值拟合函数1
    func = interpolate.interp1d(x_new, normalization(largest_components_list), kind='cubic')
    # 得到新的纵坐标
    new_largest_components_list = func(x_new)
    # 插值拟合函数2
    func2 = interpolate.interp1d(x_new, normalization(not_fail_nodes), kind='cubic')
    new_not_fail_nodes = func2(x_new)
    plt.plot(x_new, new_largest_components_list, color="red", label="最大联通子图节点数")
    plt.plot(x_new, new_not_fail_nodes, color="blue", label="未失效节点数")
    plt.ylabel("节点数占比")
    plt.title("网络级联失效图a=" + argument_a + ",w=" + argument_w)
    plt.xlabel("步长")
    plt.legend()
    plt.show()

    # 删除当前参数下的load capacity信息
    delete_info('capacity.txt')
    delete_info('load.txt')
    delete_info('G.txt')
    # 写入excel文件
    # not_fail_excel = [3, 7]
    # largest_components_excel = [3, 8]
    # data_write('data/（1）w=0.5随a变化的结果.xlsx', not_fail_nodes, 'a=1.0,w=0.5',
    #            not_fail_excel[0], not_fail_excel[1])
    # data_write('data/（1）w=0.5随a变化的结果.xlsx', largest_components_list, 'a=1.0,w=0.5',
    #            largest_components_excel[0], largest_components_excel[1])

