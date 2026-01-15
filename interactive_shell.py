#!/usr/bin/env python3
"""
Interactive Shell for Project Recommendations.

A rich, menu-driven interface for exploring project recommendations.

Usage:
    python interactive_shell.py
"""

import os
import sys
from typing import Optional, List, Dict, Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.markdown import Markdown
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better UI: pip install rich")

from recommender import ProjectRecommender


class InteractiveShell:
    """Interactive shell for project recommendations."""
    
    def __init__(self):
        """Initialize the interactive shell."""
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        self.recommender: Optional[ProjectRecommender] = None
        self._load_recommender()
    
    def _load_recommender(self):
        """Load the recommender engine."""
        try:
            self.recommender = ProjectRecommender()
            self._print_success("Recommendation engine loaded successfully!")
        except FileNotFoundError as e:
            self._print_error(str(e))
            self._print_info("Run 'python build_pipeline.py' first to build artifacts.")
            sys.exit(1)
    
    def _print(self, text: str):
        """Print text."""
        if self.console:
            self.console.print(text)
        else:
            print(text)
    
    def _print_error(self, text: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[bold red]Error:[/bold red] {text}")
        else:
            print(f"Error: {text}")
    
    def _print_success(self, text: str):
        """Print success message."""
        if self.console:
            self.console.print(f"[bold green]✓[/bold green] {text}")
        else:
            print(f"✓ {text}")
    
    def _print_info(self, text: str):
        """Print info message."""
        if self.console:
            self.console.print(f"[bold blue]ℹ[/bold blue] {text}")
        else:
            print(f"ℹ {text}")
    
    def _clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_banner(self):
        """Show application banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║         PROJECT RECOMMENDATION SYSTEM                        ║
║         Interactive Terminal Interface                       ║
╚══════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(Panel(banner.strip(), style="bold cyan"))
        else:
            print(banner)
    
    def _show_menu(self):
        """Show main menu."""
        menu = """
[bold]MAIN MENU[/bold]

  [cyan]1[/cyan] - Search by Keywords
  [cyan]2[/cyan] - Search with Filters
  [cyan]3[/cyan] - Find Similar Projects
  [cyan]4[/cyan] - Browse by Domain
  [cyan]5[/cyan] - Browse by Language
  [cyan]6[/cyan] - Browse by Difficulty
  [cyan]7[/cyan] - View Project Details
  [cyan]0[/cyan] - Exit

        """
        if self.console:
            self.console.print(menu)
        else:
            print("""
MAIN MENU

  1 - Search by Keywords
  2 - Search with Filters
  3 - Find Similar Projects
  4 - Browse by Domain
  5 - Browse by Language
  6 - Browse by Difficulty
  7 - View Project Details
  0 - Exit
            """)
    
    def _display_results(self, results: List[Dict[str, Any]], show_details: bool = False):
        """Display search results."""
        if not results:
            self._print_info("No results found.")
            return
        
        if self.console and RICH_AVAILABLE:
            table = Table(title="Search Results", show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim", width=4)
            table.add_column("ID", width=6)
            table.add_column("Title", width=35)
            table.add_column("Domain", width=20)
            table.add_column("Language", width=12)
            table.add_column("Score", width=8)
            
            for i, r in enumerate(results, 1):
                title = r['title'][:33] + ".." if len(r['title']) > 35 else r['title']
                domain = str(r['domain'])[:18] + ".." if len(str(r['domain'])) > 20 else str(r['domain'])
                lang = str(r['language'])[:10] + ".." if len(str(r['language'])) > 12 else str(r['language'])
                
                table.add_row(
                    str(i),
                    str(r['project_id']),
                    title,
                    domain,
                    lang,
                    f"{r['score']:.3f}"
                )
            
            self.console.print(table)
            
            if show_details:
                self._print("\n[dim]Enter result # to see details, or press Enter to continue[/dim]")
        else:
            print("\nSearch Results:")
            print("-" * 80)
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['project_id']}] {r['title'][:50]} - Score: {r['score']:.3f}")
            print("-" * 80)
    
    def _display_project_detail(self, project: Dict[str, Any]):
        """Display detailed project information."""
        if self.console and RICH_AVAILABLE:
            detail = f"""
## {project['title']}

**Project ID:** {project['project_id']}
**Domain:** {project['domain']}
**Language:** {project['language']}
**Difficulty:** {project['difficulty']}
**Build Time:** {project['build_time']}

### Description
{project['description']}

### Keywords
{project['keywords']}

### Libraries/Toolkits
{project['libraries']}
            """
            self.console.print(Panel(Markdown(detail), title="Project Details", border_style="green"))
        else:
            print(f"\n{'='*60}")
            print(f"Title: {project['title']}")
            print(f"Project ID: {project['project_id']}")
            print(f"Domain: {project['domain']}")
            print(f"Language: {project['language']}")
            print(f"Difficulty: {project['difficulty']}")
            print(f"Build Time: {project['build_time']}")
            print(f"\nDescription:\n{project['description']}")
            print(f"\nKeywords: {project['keywords']}")
            print(f"Libraries: {project['libraries']}")
            print(f"{'='*60}")
    
    def _prompt(self, text: str, default: str = "") -> str:
        """Get user input."""
        if self.console and RICH_AVAILABLE:
            return Prompt.ask(text, default=default)
        else:
            result = input(f"{text} [{default}]: ").strip()
            return result if result else default
    
    def _prompt_int(self, text: str, default: int = 10) -> int:
        """Get integer input."""
        if self.console and RICH_AVAILABLE:
            return IntPrompt.ask(text, default=default)
        else:
            try:
                result = input(f"{text} [{default}]: ").strip()
                return int(result) if result else default
            except ValueError:
                return default
    
    # ================== MENU ACTIONS ==================
    
    def action_search_keywords(self):
        """Search by keywords."""
        self._print("\n[bold]KEYWORD SEARCH[/bold]\n")
        
        query = self._prompt("Enter search keywords")
        if not query:
            return
        
        top_n = self._prompt_int("Number of results", 10)
        
        results = self.recommender.recommend(query=query, top_n=top_n)
        self._display_results(results, show_details=True)
        
        self._prompt("\nPress Enter to continue...")
    
    def action_search_filtered(self):
        """Search with filters."""
        self._print("\n[bold]FILTERED SEARCH[/bold]\n")
        
        query = self._prompt("Enter search keywords")
        domain = self._prompt("Filter by domain (leave empty for all)")
        language = self._prompt("Filter by language (leave empty for all)")
        difficulty = self._prompt("Filter by difficulty (leave empty for all)")
        top_n = self._prompt_int("Number of results", 10)
        
        results = self.recommender.recommend(
            query=query or "project",
            top_n=top_n,
            domain_filter=domain if domain else None,
            language_filter=language if language else None,
            difficulty_filter=difficulty if difficulty else None
        )
        self._display_results(results, show_details=True)
        
        self._prompt("\nPress Enter to continue...")
    
    def action_find_similar(self):
        """Find similar projects."""
        self._print("\n[bold]FIND SIMILAR PROJECTS[/bold]\n")
        
        project_id = self._prompt_int("Enter Project ID", 1)
        top_n = self._prompt_int("Number of similar projects", 5)
        
        try:
            results = self.recommender.recommend_by_project_id(project_id, top_n=top_n)
            self._print(f"\nProjects similar to ID {project_id}:\n")
            self._display_results(results)
        except ValueError as e:
            self._print_error(str(e))
        
        self._prompt("\nPress Enter to continue...")
    
    def action_browse_domain(self):
        """Browse by domain."""
        self._print("\n[bold]BROWSE BY DOMAIN[/bold]\n")
        
        domains = self.recommender.get_domains()
        for i, d in enumerate(domains, 1):
            self._print(f"  {i}. {d}")
        
        choice = self._prompt("\nSelect domain number or enter domain name")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(domains):
                domain = domains[idx]
            else:
                domain = choice
        except ValueError:
            domain = choice
        
        results = self.recommender.recommend(
            query="project",
            domain_filter=domain,
            top_n=20
        )
        self._print(f"\nProjects in domain '{domain}':\n")
        self._display_results(results)
        
        self._prompt("\nPress Enter to continue...")
    
    def action_browse_language(self):
        """Browse by programming language."""
        self._print("\n[bold]BROWSE BY LANGUAGE[/bold]\n")
        
        languages = self.recommender.get_languages()
        for i, l in enumerate(languages, 1):
            self._print(f"  {i}. {l}")
        
        choice = self._prompt("\nSelect language number or enter language name")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(languages):
                language = languages[idx]
            else:
                language = choice
        except ValueError:
            language = choice
        
        results = self.recommender.recommend(
            query="project",
            language_filter=language,
            top_n=20
        )
        self._print(f"\nProjects using '{language}':\n")
        self._display_results(results)
        
        self._prompt("\nPress Enter to continue...")
    
    def action_browse_difficulty(self):
        """Browse by difficulty level."""
        self._print("\n[bold]BROWSE BY DIFFICULTY[/bold]\n")
        
        levels = self.recommender.get_difficulty_levels()
        for i, l in enumerate(levels, 1):
            self._print(f"  {i}. {l}")
        
        choice = self._prompt("\nSelect difficulty number or enter level name")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(levels):
                level = levels[idx]
            else:
                level = choice
        except ValueError:
            level = choice
        
        results = self.recommender.recommend(
            query="project",
            difficulty_filter=level,
            top_n=20
        )
        self._print(f"\nProjects at '{level}' level:\n")
        self._display_results(results)
        
        self._prompt("\nPress Enter to continue...")
    
    def action_view_details(self):
        """View project details by ID."""
        self._print("\n[bold]VIEW PROJECT DETAILS[/bold]\n")
        
        project_id = self._prompt_int("Enter Project ID", 1)
        
        # Get project by searching for it
        results = self.recommender.recommend(
            query="project",
            top_n=1000
        )
        
        project = None
        for r in results:
            if r['project_id'] == project_id:
                project = r
                break
        
        if project:
            self._display_project_detail(project)
        else:
            self._print_error(f"Project ID {project_id} not found.")
        
        self._prompt("\nPress Enter to continue...")
    
    def run(self):
        """Run the interactive shell."""
        self._clear_screen()
        self._show_banner()
        
        while True:
            self._show_menu()
            
            choice = self._prompt("Select option")
            
            if choice == "0":
                self._print("\n[bold green]Goodbye![/bold green]\n")
                break
            elif choice == "1":
                self.action_search_keywords()
            elif choice == "2":
                self.action_search_filtered()
            elif choice == "3":
                self.action_find_similar()
            elif choice == "4":
                self.action_browse_domain()
            elif choice == "5":
                self.action_browse_language()
            elif choice == "6":
                self.action_browse_difficulty()
            elif choice == "7":
                self.action_view_details()
            else:
                self._print_error("Invalid option. Please try again.")
            
            self._clear_screen()
            self._show_banner()


if __name__ == "__main__":
    shell = InteractiveShell()
    shell.run()
