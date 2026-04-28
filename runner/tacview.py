import socket # 用于网络通信，创建 TCP 服务器和客户端
import atexit # 注册退出时的清理函数，保证资源释放
import signal # 捕获信号（如 Ctrl+C），优雅退出
import sys 

class Tacview(object):
    def __init__(self):
        atexit.register(self.cleanup)  # 注册退出清理
        #signal.signal(signal.SIGTSTP, self.handle_sigtstp)  # 捕获 Ctrl+Z
        signal.signal(signal.SIGINT, self.handle_sigint)  # 捕获 Ctrl+C
        # 确保程序在：正常退出、Ctrl+C 终止、Ctrl+Z 挂起、异常退出都能正确释放资源
        # Automatically get the local machine's IP address
        self.host = self.get_ip_address() # 获取本机IP地址，用于Tacview客户端填写
        # Default starting port
        self.port = 12345 # 默认端口
        self.setup_server() #启动服务器
    
    def handle_sigtstp(self, signum, frame):
        """ 处理 Ctrl+Z 信号 """
        print("\n捕获到 Ctrl+Z，正在清理资源...")
        self.cleanup()
        sys.exit(0)  # 退出程序

    def handle_sigint(self, signum, frame):
        """ 处理 Ctrl+C 信号 """
        print("\n捕获到 Ctrl+C，正在清理资源...")
        self.cleanup()
        sys.exit(0)  # 退出程序

    def get_ip_address(self): # 获取本机IP地址，用于Tacview客户端填写
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))  # Google DNS, 只是为了获得正确的IP地址
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address

    """
    def get_ip_address(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # 连接到一个不会真的被访问的局域网地址
            s.connect(('10.255.255.255', 1))
            ip_address = s.getsockname()[0]
        except Exception:
            ip_address = '127.0.0.1'
        finally:
            s.close()
        return ip_address
    """
    
    def setup_server(self): # 启动服务器
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 获取tcp/ip套接字
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 设置socket选项，允许重用本地地址和端口
            self.server_socket.bind((self.host, self.port)) # 绑定IP地址和端口
            self.server_socket.listen(5) # 监听连接
            print(f"Server listening on {self.host}:{self.port}")
            # Output more prominent message
            print("\n" + "*" * 50)
            print("! IMPORTANT: Please open Tacview Advanced, click Record -> Real-time Telemetry, and input the IP address and port !")
            print("*" * 50 + "\n")
            self.connect()
        except Exception as e:
            print(f"Setup error: {e}")
            self.cleanup()
            raise

    def send_data_to_client(self, data): # 发送数据到客户端
        try:
            self.client_socket.send(data.encode()) # 通过已建立的客户端套接字发送数据。data.encode() 将字符串转换为字节序列（bytes）
        except Exception as e:
            print(f"Send error: {e}")
            self.reconnect()

    """
    try:
        # 可能会引发异常的代码
    except Exception as e:
        # 处理异常的代码
    如果执行try块中的代码出现异常,程序立刻跳转到except块中,并将异常对象赋值给变量e
    """
            
    def connect(self):
        try:
            print("Waiting for connection...")
            self.client_socket, self.address = self.server_socket.accept()
            print(f"Accepted connection from {self.address}")
            
            # 发送握手数据
            handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake_data.encode())
            
            # 接收客户端响应
            data = self.client_socket.recv(1024)
            print(f"Received data from {self.address}: {data.decode()}")
            
            # 发送头部数据
            header_data = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                          "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n")
            self.client_socket.send(header_data.encode())
            print("Connection established")
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.cleanup()
            raise

    def reconnect(self):
        print("Attempting to reconnect...")
        self.cleanup()
        self.setup_server()

    def cleanup(self):
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            if hasattr(self, 'server_socket') and self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        self.cleanup()