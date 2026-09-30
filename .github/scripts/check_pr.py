"""PR의 Issue 연결과 커밋 제목 규칙을 검사한다."""
import os
import re
import subprocess
import sys

PATTERN = re.compile(r"^(feat|fix|docs|refactor|test|chore|ci|build|perf): .*[가-힣].*$")
base = os.environ["BASE_SHA"]
head = os.environ["HEAD_SHA"]
title = os.environ["PR_TITLE"]
body = os.environ.get("PR_BODY", "")
repo = os.environ["GITHUB_REPOSITORY"]
issue_link = re.compile(r"(?:^|\s)(?:#\d+|https://github\.com/" + re.escape(repo) + r"/issues/\d+)(?:\b|$)")
errors = []
if not PATTERN.fullmatch(title):
    errors.append("PR 제목은 'feat: 한글 설명' 형식이어야 합니다.")
if not issue_link.search(body):
    errors.append("PR 본문에 같은 저장소의 Issue 번호(예: Closes #1)를 적어야 합니다.")
commits = subprocess.check_output(
    ["git", "log", "--format=%s", f"{base}..{head}"], text=True
).splitlines()
if not commits:
    errors.append("검사할 커밋이 없습니다.")
for title_line in commits:
    if not PATTERN.fullmatch(title_line):
        errors.append(f"커밋 제목 형식 오류: {title_line}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"PR과 커밋 {len(commits)}개의 규칙을 확인했습니다.")
