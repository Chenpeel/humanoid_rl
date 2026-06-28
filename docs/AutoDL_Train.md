

## 创建实例

- 选择conda环境， 任意都可以，只是为了安装uv
- 选择RTX系列显卡

## 登入远程主机

> 使用AutoDL提供的SSH方式登入

#### 配置Vulkan

1. 写入ICD

    ```bash
    cat >> /etc/vulkan/icd.d/my_nvidia_icd.json <<EOF
    {
        "file_format_version" : "1.0.0",
        "ICD": {
            "library_path": "/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
            "api_version" : "1.3.277"
        }
    }
    EOF
    ```

2. 安装依赖

    ```bash
    apt update && apt install -y vulkan-tools libvulkan1 libsm6 libegl1
    ```

3. 写入环境变量

    ```bash
    echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json" >> ~/.bashrc
    source ~/.bashrc
    ```

4. 验证

    ```bash
    vulkaninfo --summary
    ```

#### 使用conda/pip安装uv

```bash
# 使用 pip 安装 uv
pip install uv

# 或者使用 conda
# conda install -c conda-forge uv

uv --version
```



#### 克隆本库

1. 切换到数据盘

    ```bash
    cd autodl-tmp/
    ```

2. 克隆本库

    ```bash
    git clone https://github.com/Chenpeel/rl.git
    ```

3. 按照本库的README执行
