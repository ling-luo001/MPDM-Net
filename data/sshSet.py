# 1. 安装并启动 SSH 服务
# 更新软件包列表并安装 OpenSSH 服务端：
#
# Bash
# sudo apt update
# sudo apt install openssh-server -y
# 安装完成后，检查 SSH 服务的运行状态：
#
# Bash
# sudo systemctl status ssh
# 验证标准：如果看到绿色的 active (running) 字样，说明服务已成功启动。按 q 键退出状态查看。
#
# 2. 配置防火墙（如果已启用）
# 确保 Ubuntu 的防火墙允许 SSH 连接（即放行 22 端口）：
#
# Bash
# sudo ufw allow ssh
# 3. 获取 Ubuntu 的 IP 地址
# 输入以下命令查看网络信息：
#
# Bash
# ip a
# 验证标准：在输出信息中寻找你当前连接的网卡（通常是有线网卡 enp... 或无线网卡 wlan... 或 wlp...），找到 inet 后面跟着的 IP 地址（例如 192.168.x.x）。请将这个 IP 地址、以及您登录这台 Ubuntu 的用户名和开机密码记录下来备用。