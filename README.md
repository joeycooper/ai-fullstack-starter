# AI Fullstack Starter

一个本地优先的「上传上下文 + RAG 检索 + 对话」示例项目。前端为 React/Vite，后端为 FastAPI + SQLite，默认支持本地向量化回退，也可接入 DeepSeek（OpenAI 兼容 API）生成回答。

## 功能概览

- 上传文本并切分为段落，进行向量化存储
- 基于向量检索（余弦相似度）找出相关上下文
- 支持全量检索 / 仅最新批次检索
- 支持本地 embedding 回退与在线模型生成
- 前端展示命中段落与匹配度

## 快速启动（本地开发）

### 1) 后端

在项目根目录准备环境配置：

```
cp .env.example .env
```

如果暂时没有 DeepSeek Key，可以使用本地回退：

```
USE_LOCAL_EMBEDDINGS_FALLBACK=1
DEEPSEEK_API_KEY=
```

启动后端（在 backend 目录）：

```
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

### 2) 前端

```
cd frontend
npm install
npm run dev
```

前端默认使用 `/api`，Vite 代理已配置为转发到 `http://127.0.0.1:8000`。

## 临时公网访问（可选）

使用 cloudflared 一键启动：

```
/Users/joey/work/ai-fullstack-starter/start-dev-tunnel.sh
```

脚本会自动打开本地预览和 cloudflared 公网地址。

## 环境变量说明（节选）

- `DEEPSEEK_API_KEY`：DeepSeek API Key（为空则不调用在线模型）
- `USE_LOCAL_EMBEDDINGS_FALLBACK`：无 Key 时启用本地向量化
- `USE_ALL_CONTEXTS`：1 为检索所有历史内容，0 仅检索最新批次
- `TOP_K`：检索段落数量
- `MAX_CONTEXT_CHARS`：拼接到 prompt 的最大字符数

完整配置见 `.env.example` / `.env.prod`。

## 后续可优化方向

- 嵌入模型优化：切换更适合中文的向量模型
- 检索策略：引入 BM25 / 混合检索、加入重排序
- 批次管理：支持选择/删除某次上传的上下文
- 权限与安全：上传大小限制、简单鉴权、日志与监控
- 部署优化：Nginx + HTTPS、Docker 化、CI/CD 自动化
