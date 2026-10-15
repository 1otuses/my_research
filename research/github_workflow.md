# GitHub 阶段工作流

目标仓库为公开的 `1otuses/my_research`，已由用户明确选择。该仓库追踪研究文档；算法代码继续在 `1otuses/madiff` 的研究分支推进，两者用固定 commit 和 PR 相互关联。

远端仓库：[1otuses/my_research](https://github.com/1otuses/my_research)。用户于 2026-10-15 创建公开仓库，初始研究文档已通过本机 Git 推送到 `main`。

每项工作完成后：更新对应研究文档和日志，记录一手来源或代码/数据指纹，检查文件和结论，提交到 `codex/<阶段>-<任务>`，push 并创建 PR。初始化研究仓库可先建立 `main` 的第一笔文档提交；后续工作通过分支和 PR 保留讨论记录。

阶段日志区分“计划”“已运行”“结果支持”“结果不支持”。论文清单记录版本，实验记录固定代码 commit，负结果也提交。研究方向改变时另写决策记录，并在 README 更新当前主线。

数据、checkpoint 和大型中间产物存于适合的外部存储，在仓库记录来源、路径约定和校验值。论文 PDF 通过来源索引关联，避免把整个本地资料库随初始化提交。

## 上传与同步

本机 CLI 凭据与 GitHub 连接器认证相互独立。本轮连接器写入返回 `403 Resource not accessible by integration`，应用安装与账户列表为空；该连接器的写入能力尚未恢复。

本机已有 Git Credential Manager，通过以下浏览器登录命令完成认证后，实际推送成功：

```bash
git credential-manager github login --username 1otuses --browser
git push -u origin main
```

后续上传优先使用已验证的本机 Git 登录。提交前检查修改范围，推送后获取远端状态并核对提交 SHA。初始化来源 `f36f7754c0efe6c91fc24ac4884bfd2cea4777a2` 完整保留在 `main` 历史中。

常规阶段更新可执行：

```bash
git switch -c codex/<阶段>-<任务>
git add <本阶段文件>
git commit -m "docs: record research milestone"
git push -u origin HEAD
```

随后在 GitHub 创建 PR，记录问题、证据、修改和验证。如需通过连接器创建 PR，应先核实连接器实际写入权限；已经创建的 PR 关联到当前 Codex 聊天。
