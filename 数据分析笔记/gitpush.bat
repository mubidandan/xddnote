title GIT提交批处理
@echo off
::git config --global user.name  "username"  
::git config --global user.email  "email"

echo 开始提交代码到本地仓库
echo 当前目录是：%cd%

echo 开始添加变更
echo ------------------------------------------------------------
git add -A .
echo 执行结束！
echo ------------------------------------------------------------
echo;
echo 提交变更到本地仓库
echo ------------------------------------------------------------
set /p declation=输入修改:
git commit -m "%declation%"
echo ------------------------------------------------------------
echo;
echo 将变更情况提交到远程git服务器
echo ------------------------------------------------------------
git push mayun master
git push origin master
echo ------------------------------------------------------------
echo;
echo 代码上传完成
