#!/usr/bin/env python3
"""
================================================================================
Nemotron-Indonesia ↔ Multica Project Sync
================================================================================

Sync Nemotron training pipeline progress to your Multica project management
instance. Creates tasks, updates status, and links artifacts automatically.

Usage:
    # Setup: Create project + tasks in Multica
    python multica_sync.py --setup --api-token YOUR_TOKEN --workspace-id YOUR_WS

    # Update progress after completing a phase
    python multica_sync.py --update-phase data_prep --status done

    # Full sync (reads local progress files + pushes to Multica)
    python multica_sync.py --sync

Required:
    - Multica API token (get from Multica UI → Settings → API Tokens)
    - Multica workspace ID (from URL: /workspace/WORKSPACE_ID)
    - Multica server URL (default: https://multica.jatevo.ai)

Environment variables (optional):
    MULTICA_API_TOKEN=your_token
    MULTICA_WORKSPACE_ID=your_workspace
    MULTICA_URL=https://multica.jatevo.ai
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import urllib.request
import urllib.error


class MulticaClient:
    """Simple REST client for Multica self-hosted API."""

    def __init__(self, base_url: str, token: str, workspace_id: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.workspace_id = workspace_id

    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Workspace-ID": self.workspace_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            print(f"❌ API Error {e.code}: {error_body}")
            raise
        except Exception as e:
            print(f"❌ Request failed: {e}")
            raise

    def get(self, path: str) -> Dict:
        return self._request("GET", path)

    def post(self, path: str, body: Dict) -> Dict:
        return self._request("POST", path, body)

    def patch(self, path: str, body: Dict) -> Dict:
        return self._request("PATCH", path, body)


class NemotronProjectSync:
    """Sync Nemotron pipeline progress to Multica."""

    PHASES = [
        {
            "id": "data_prep",
            "title": "Phase 1: Data Preparation",
            "description": "Download, clean, quality-score, deduplicate, and tokenize 20B Indonesian tokens",
            "tasks": [
                "Download OSCAR, CC100, Wikipedia, Liputan6",
                "Clean text (regex, length, language filters)",
                "NER quality scoring (cahya/bert-base-indonesian-NER)",
                "Deduplicate with MinHash LSH",
                "Tokenize and package final corpus",
            ],
            "estimated_hours": 2,
            "github_file": "prepare_data.py",
            "github_url": "https://github.com/lucacadalora/nemotron-indonesia/blob/main/prepare_data.py",
        },
        {
            "id": "pretrain",
            "title": "Phase 2: Continued Pre-Training",
            "description": "Continue pre-training Nemotron 30B on 20B Indonesian tokens for 3 days",
            "tasks": [
                "Configure DeepSpeed ZeRO-3 on 8× H200",
                "Run continued pre-training (20B tokens, 3 days)",
                "Monitor loss curves, save checkpoints every 500 steps",
                "Validate intermediate checkpoints",
            ],
            "estimated_hours": 72,
            "github_file": "train_nemotron_indonesia.py",
            "github_url": "https://github.com/lucacadalora/nemotron-indonesia/blob/main/train_nemotron_indonesia.py",
        },
        {
            "id": "sft",
            "title": "Phase 3: Supervised Fine-Tuning",
            "description": "Fine-tune on 500K instruction pairs for chat/agentic capabilities",
            "tasks": [
                "Prepare SFT dataset (IndoMMLU + NusaX + custom)",
                "Run SFT training (~12 hours)",
                "Evaluate on IndoMMLU validation set",
            ],
            "estimated_hours": 12,
            "github_file": "train_nemotron_indonesia.py",
            "github_url": "https://github.com/lucacadalora/nemotron-indonesia/blob/main/train_nemotron_indonesia.py",
        },
        {
            "id": "dpo",
            "title": "Phase 4: DPO Alignment",
            "description": "Align model with human preferences using DPO",
            "tasks": [
                "Prepare 50K preference pairs (chosen vs rejected)",
                "Run DPO training (~6 hours)",
                "Final evaluation on full benchmark suite",
                "Red-teaming: spell-check, proper nouns, human panel",
            ],
            "estimated_hours": 8,
            "github_file": "train_nemotron_indonesia.py",
            "github_url": "https://github.com/lucacadalora/nemotron-indonesia/blob/main/train_nemotron_indonesia.py",
        },
        {
            "id": "deploy",
            "title": "Phase 5: Deployment & Launch",
            "description": "Push to HuggingFace, deploy API, benchmark vs Sahabat AI",
            "tasks": [
                "Push model to HuggingFace (jatevo/nemotron-indonesia-30B)",
                "Deploy inference API on Jatevo Vortex",
                "Write launch blog post / announcement",
                "Benchmark against Sahabat AI 70B",
            ],
            "estimated_hours": 4,
            "github_file": "README.md",
            "github_url": "https://github.com/lucacadalora/nemotron-indonesia",
        },
    ]

    def __init__(self, client: MulticaClient, project_id: Optional[str] = None):
        self.client = client
        self.project_id = project_id
        self.status_file = Path(__file__).parent / "nemotron_status.json"

    def load_status(self) -> Dict:
        """Load local Nemotron progress status."""
        if self.status_file.exists():
            with open(self.status_file) as f:
                return json.load(f)
        return {
            "phases": {p["id"]: {"status": "not_started", "progress": 0} for p in self.PHASES},
            "current_phase": None,
            "last_updated": None,
        }

    def save_status(self, status: Dict):
        """Save local progress status."""
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def find_or_create_project(self) -> str:
        """Find existing Nemotron project or create new one."""
        # Try to find existing
        try:
            projects = self.client.get("/api/projects")
            if isinstance(projects, list):
                for p in projects:
                    if p.get("title", "").lower() in ["nemotron-indonesia", "nemotron indonesia"]:
                        print(f"✅ Found existing project: {p['id']}")
                        return p["id"]
            elif isinstance(projects, dict) and "data" in projects:
                for p in projects["data"]:
                    if p.get("title", "").lower() in ["nemotron-indonesia", "nemotron indonesia"]:
                        print(f"✅ Found existing project: {p['id']}")
                        return p["id"]
        except Exception as e:
            print(f"⚠️ Could not list projects: {e}")

        # Create new project
        print("🆕 Creating Nemotron-Indonesia project...")
        resp = self.client.post("/api/projects", {
            "title": "Nemotron-Indonesia",
            "description": "Indonesian sovereign 30B LLM built on NVIDIA Nemotron — agentic AI for Bahasa Indonesia and 10+ local languages. Beat Sahabat AI 70B on IndoMMLU.",
            "status": "active",
            "priority": "high",
            "icon": "🤖",
        })
        project_id = resp.get("id") or resp.get("data", {}).get("id")
        print(f"✅ Created project: {project_id}")
        return project_id

    def create_phase_issues(self, project_id: str):
        """Create issues for each pipeline phase."""
        for phase in self.PHASES:
            issue_title = phase["title"]

            # Check if issue already exists
            try:
                issues = self.client.get(f"/api/projects/{project_id}/issues")
                existing = [i for i in (issues.get("data", []) if isinstance(issues, dict) else issues)
                           if i.get("title") == issue_title]
                if existing:
                    print(f"⏭️ Issue exists: {issue_title}")
                    continue
            except Exception:
                pass

            # Create issue
            body = f"""## {phase['description']}

### Tasks
{chr(10).join(f"- [ ] {t}" for t in phase['tasks'])}

### Key File
- [{phase['github_file']}]({phase['github_url']})

### Estimated Time
{phase['estimated_hours']} hours

---
*Generated by Nemotron-Multica sync*"""

            print(f"📝 Creating issue: {issue_title}")
            self.client.post("/api/issues", {
                "project_id": project_id,
                "title": issue_title,
                "body": body,
                "status": "backlog",
                "priority": "high",
                "labels": ["nemotron", phase["id"], "ml-training"],
            })

    def update_phase_status(self, phase_id: str, status: str, progress: Optional[int] = None):
        """Update a phase's status locally and in Multica."""
        local_status = self.load_status()
        local_status["phases"][phase_id]["status"] = status
        if progress is not None:
            local_status["phases"][phase_id]["progress"] = progress
        local_status["last_updated"] = json.dumps({})
        self.save_status(local_status)

        if self.project_id:
            # Find and update corresponding issue in Multica
            try:
                phase = next(p for p in self.PHASES if p["id"] == phase_id)
                issues = self.client.get(f"/api/projects/{self.project_id}/issues")
                issue_list = issues.get("data", []) if isinstance(issues, dict) else issues
                for issue in issue_list:
                    if issue.get("title") == phase["title"]:
                        multica_status = {
                            "not_started": "backlog",
                            "in_progress": "in_progress",
                            "done": "done",
                            "blocked": "backlog",
                        }.get(status, "backlog")

                        self.client.patch(f"/api/issues/{issue['id']}", {
                            "status": multica_status,
                        })
                        print(f"✅ Updated Multica: {phase['title']} → {multica_status}")
                        break
            except Exception as e:
                print(f"⚠️ Could not update Multica: {e}")

    def print_progress_dashboard(self):
        """Print a CLI progress dashboard."""
        status = self.load_status()
        print("\n" + "=" * 70)
        print("NEMOTRON-INDONESIA PIPELINE PROGRESS")
        print("=" * 70)

        for phase in self.PHASES:
            st = status["phases"].get(phase["id"], {})
            status_str = st.get("status", "not_started")
            progress = st.get("progress", 0)

            icons = {
                "not_started": "⬜",
                "in_progress": "🟡",
                "done": "✅",
                "blocked": "🔴",
            }
            icon = icons.get(status_str, "⬜")

            bar = "█" * (progress // 5) + "░" * (20 - progress // 5)
            print(f"{icon} {phase['title']}")
            print(f"   [{bar}] {progress}% | {status_str}")
            print()

        print("=" * 70)
        print(f"Current: {status.get('current_phase', 'None')}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sync Nemotron progress to Multica")
    parser.add_argument("--setup", action="store_true", help="Create project + issues in Multica")
    parser.add_argument("--sync", action="store_true", help="Sync local progress to Multica")
    parser.add_argument("--update-phase", type=str, help="Update phase status (data_prep|pretrain|sft|dpo|deploy)")
    parser.add_argument("--status", type=str, default="done", choices=["not_started", "in_progress", "done", "blocked"])
    parser.add_argument("--progress", type=int, help="Progress percentage (0-100)")
    parser.add_argument("--api-token", type=str, default=os.environ.get("MULTICA_API_TOKEN"))
    parser.add_argument("--workspace-id", type=str, default=os.environ.get("MULTICA_WORKSPACE_ID"))
    parser.add_argument("--multica-url", type=str, default=os.environ.get("MULTICA_URL", "https://multica.jatevo.ai"))
    parser.add_argument("--project-id", type=str, help="Existing Multica project ID")
    parser.add_argument("--dashboard", action="store_true", help="Show progress dashboard")

    args = parser.parse_args()

    if args.dashboard:
        sync = NemotronProjectSync(None)
        sync.print_progress_dashboard()
        return

    if not args.api_token or not args.workspace_id:
        print("❌ Missing API token or workspace ID")
        print("\nGet your token from Multica UI → Settings → API Tokens")
        print("Get workspace ID from URL: /workspace/WORKSPACE_ID")
        print("\nSet env vars:")
        print("  export MULTICA_API_TOKEN=your_token")
        print("  export MULTICA_WORKSPACE_ID=your_workspace")
        sys.exit(1)

    client = MulticaClient(args.multica_url, args.api_token, args.workspace_id)
    sync = NemotronProjectSync(client, args.project_id)

    if args.setup:
        project_id = sync.find_or_create_project()
        sync.project_id = project_id
        sync.create_phase_issues(project_id)
        print("\n✅ Setup complete! Project and issues created in Multica.")
        print(f"   Project ID: {project_id}")
        print(f"   URL: {args.multica_url}/project/{project_id}")
        return

    if args.update_phase:
        if not sync.project_id:
            print("❌ Need --project-id or run --setup first")
            sys.exit(1)
        sync.update_phase_status(args.update_phase, args.status, args.progress)
        print(f"✅ Updated phase: {args.update_phase} → {args.status}")
        return

    if args.sync:
        if not sync.project_id:
            print("❌ Need --project-id or run --setup first")
            sys.exit(1)
        # Full sync logic
        sync.print_progress_dashboard()
        print("\n🔄 Full sync not yet implemented. Use --update-phase for now.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
