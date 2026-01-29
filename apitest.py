from src.utils.utils import load_config, save_data
from github import Github
import github.Auth
import json
CONFIG = load_config()


access_token = CONFIG.get("token")
repo_owner = CONFIG.get("owner")
repo_name = CONFIG.get("repo")

auth = github.Auth.Token(access_token)
g = Github(auth=auth)

repo = g.get_repo(f"{repo_owner}/{repo_name}")
print(repo)
save_data(repo.raw_data, "repo_test.json")

issues = repo.get_issues(state="all")  ## 获取所有状态的issue,不加state参数默认只获取open状态的issue
save_data([issue.raw_data for issue in issues], "issues_test.json")
issue = repo.get_issue(number=1)
commits = repo.get_commits()
save_data([commit.raw_data for commit in commits], "commits_test.json")
events = issue.get_events()
save_data([event.raw_data for event in events], "events_test.json")