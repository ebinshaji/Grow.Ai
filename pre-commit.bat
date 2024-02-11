@echo off
rem Automatically add all changes and commit them
git add .
git commit -m "Automatic commit before pushing to remote repository"

rem Push changes to the remote repository
git push origin HEAD
