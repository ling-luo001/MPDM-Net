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
#
# 127.0.0.1/8

# (base) lz@lz-System-Product-Name:~$ ip a
# 1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
#     link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
#     inet 127.0.0.1/8 scope host lo
#        valid_lft forever preferred_lft forever
#     inet6 ::1/128 scope host
#        valid_lft forever preferred_lft forever
# 2: enp4s0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
#     link/ether 04:42:1a:29:0c:99 brd ff:ff:ff:ff:ff:ff
# 3: wlp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
#     link/ether c0:3c:59:ba:68:b4 brd ff:ff:ff:ff:ff:ff
#     inet 192.168.123.186/24 brd 192.168.123.255 scope global dynamic noprefixroute wlp5s0
#        valid_lft 76065sec preferred_lft 76065sec
#     inet6 fe80::27e1:a8ac:64fa:cb3e/64 scope link noprefixroute
#        valid_lft forever preferred_lft forever
# 验证标准：在输出信息中寻找你当前连接的网卡（通常是有线网卡 enp... 或无线网卡 wlan... 或 wlp...），找到 inet 后面跟着的 IP 地址（例如 192.168.x.x）。请将这个 IP 地址、以及您登录这台 Ubuntu 的用户名和开机密码记录下来备用。