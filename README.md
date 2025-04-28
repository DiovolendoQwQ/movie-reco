# 电影推荐系统 (Movie Recommender)

本项目实现了一个基于 Item-to-Item 协同过滤算法的电影推荐系统。用户可以选择一个电影类型获取初始推荐，然后通过点击感兴趣的电影来获得更个性化的相似电影推荐。系统会记录用户最近的 5 次选择来进行推荐。

## 项目结构

```
movie-reco/
├── README.md                  # 本文件，项目说明
├── requirements.txt           # Python 依赖包列表 (用于 Docker 构建和本地开发)
├── requirements.yml           # (可能存在的 Conda 环境文件，根据用户初始设置)
├── Dockerfile                 # 用于构建 FastAPI 后端服务的 Docker 镜像配置
├── docker-compose.yml         # Docker Compose 配置，用于一键启动 Redis, API, Nginx 服务
├── nginx.conf                 # Nginx 配置文件，用于服务静态文件和反向代理 API
│
├── data/                      # 存放数据文件
│   ├── raw/                   # 存放原始 MovieLens 数据集 (ml-25m)
│   │   └── ml-25m/            # 解压后的 MovieLens 数据文件 (.csv)
│   └── processed/             # 存放预处理后的数据和模型文件
│       ├── user_item.npz      # 用户-物品交互稀疏矩阵 (CSR 格式)
│       ├── genre_map.parquet  # 电影 ID、内部索引、标题、类型的映射表
│       └── sim.parquet        # 预计算的物品(电影)相似度表 (Top-K)
│
├── scripts/                   # 存放数据处理和模型训练的 Python 脚本
│   ├── 01_download_data.py    # 下载并解压 MovieLens 数据集
│   ├── 02_build_matrix.py     # 读取评分数据，构建用户-物品交互矩阵和类型映射
│   ├── 03_compute_sim.py      # 使用 implicit 库计算物品相似度
│   └── 04_evaluate.py         # (可选) 评估推荐模型性能的脚本
│
├── app/                       # FastAPI 后端应用代码
│   ├── main.py                # FastAPI 应用入口点 (Uvicorn 使用)
│   ├── api.py                 # 定义 FastAPI 路由 (API 端点) 和 CORS 配置
│   ├── recommend.py           # 核心推荐逻辑 (加载模型、随机推荐、相似推荐)
│   ├── session.py             # 使用 Redis 管理用户会话 (电影选择历史)
│   ├── utils.py               # 通用工具函数 (如日志配置)
│   └── models/                # 存放 API 运行时加载的模型文件 (从 data/processed/ 复制而来)
│       ├── genre_map.parquet  # 电影信息映射表
│       └── sim.parquet        # 物品相似度表
│
├── web/                       # 纯静态前端页面
│   ├── index.html             # 首页，用于选择电影类型
│   ├── step.html              # 推荐结果展示页面
│   ├── step.js                # 前端 JavaScript 逻辑 (API 请求、DOM 操作)
│   └── css/                   # CSS 样式文件
│       └── tailwind.min.css   # (占位符或实际的 Tailwind CSS 文件)
│
└── tests/                     # 存放测试代码
    ├── test_recommend.py      # 推荐逻辑单元测试 (使用 pytest)
    └── test_api.py            # API 接口测试 (使用 pytest 和 httpx)

```

## 技术栈

*   **数据处理**: Polars, NumPy, SciPy
*   **相似度计算**: implicit (Item-to-Item Cosine Similarity)
*   **后端框架**: FastAPI + Uvicorn
*   **缓存/会话**: Redis
*   **前端**: 纯静态 HTML + Tailwind CSS (通过 CDN 加载) + Vanilla JavaScript
*   **容器化**: Docker + Docker Compose
*   **Web 服务器/代理**: Nginx
*   **测试**: pytest, pytest-asyncio, httpx

环境安装指令
*   conda env create -f requirements.yml
*   conda activate movie-reco
## 如何运行

1.  **前提**:
    *   安装 Docker Desktop 并确保其正在运行。
    *   (可选) 如果你想在本地运行脚本或测试，需要安装 Python 和 `requirements.txt` 中的依赖 (`pip install -r requirements.txt`)。
2.  **启动应用**: 在项目根目录下运行以下命令：
    ```bash
    docker compose up -d
    ```d
    这将构建 Docker 镜像（如果需要）并启动 Redis、API 和 Nginx 服务。
3.  **访问应用**: 在浏览器中打开 `http://localhost`。
4.  **停止应用**: 在项目根目录下运行：
    ```bash
    docker compose down
    ```

## 注意事项

*   `scripts/03_compute_sim.py` 在完整数据集上运行时可能需要较长时间。
*   前端使用了 Tailwind CSS CDN，确保网络连接正常。如果需要离线使用，请下载 Tailwind CSS 文件替换 `web/css/tailwind.min.css` 并修改 HTML 文件中的链接。
*   API 测试 (`tests/test_api.py`) 需要应用通过 `docker compose up -d` 正在运行。
