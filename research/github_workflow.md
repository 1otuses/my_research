# GitHub 阶段工作流

目标仓库为公开的 `1otuses/my_research`，已由用户明确选择。该仓库追踪研究文档；算法代码继续在 `1otuses/madiff` 的研究分支推进，两者用固定 commit 和 PR 相互关联。

远端仓库：[1otuses/my_research](https://github.com/1otuses/my_research)。用户于 2026-10-15 创建公开空仓库。读取仓库信息成功，但写入请求返回 `403 Resource not accessible by integration`，初始文档尚未上传。

每项工作完成后：更新对应研究文档和日志，记录一手来源或代码/数据指纹，检查文件和结论，提交到 `codex/<阶段>-<任务>`，push 并创建 PR。初始化研究仓库可先建立 `main` 的第一笔文档提交；后续工作通过分支和 PR 保留讨论记录。

阶段日志区分“计划”“已运行”“结果支持”“结果不支持”。论文清单记录版本，实验记录固定代码 commit，负结果也提交。研究方向改变时另写决策记录，并在 README 更新当前主线。

数据、checkpoint 和大型中间产物存于适合的外部存储，在仓库记录来源、路径约定和校验值。论文 PDF 通过来源索引关联，避免把整个本地资料库随初始化提交。

## 上传与同步

本机 CLI 凭据与 GitHub 连接器认证相互独立。CLI 未登录时，可以通过连接器上传现有仓库中的文档；连接器未提供创建仓库接口，因此由用户创建初始空仓库。

恢复连接器写入权限后，计划通过 `git fetch origin` 获取远端提交，核查文件树和内容一致，再同步本地 `main`。本地首次提交 `f36f7754c0efe6c91fc24ac4884bfd2cea4777a2` 仍保留在当前本地历史中。

仓库元数据中的 `permissions.push: true` 表示账户权限，不能证明当前连接器具备实际写入权限。本轮实际创建 README 请求被 GitHub 拒绝；GitHub 应用安装与已安装账户查询均返回空列表。本机非交互 `git push` 也因缺少登录凭据失败。

若之后配置了本机 GitHub 登录，常规阶段更新可执行：

```bash
git switch -c codex/<阶段>-<任务>
git add <本阶段文件>
git commit -m "docs: record research milestone"
git push -u origin HEAD
```

随后创建 PR，记录问题、证据、修改和验证。通过连接器创建的 PR 也关联到当前 Codex 聊天。
