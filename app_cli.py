#!/usr/bin/env python3
"""
Simple CLI for Project Recommendations.

Usage Examples:
    python app_cli.py --query "machine learning python"
    python app_cli.py --query "web development" --top 10
    python app_cli.py --query "deep learning" --domain "Computer Vision"
    python app_cli.py --query "flask api" --language "python" --difficulty "beginner"
    python app_cli.py --similar 42  # Find projects similar to project ID 42
"""

import argparse
from tabulate import tabulate
from recommender import ProjectRecommender


def main():
    parser = argparse.ArgumentParser(
        description="Project Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_cli.py --query "machine learning python"
  python app_cli.py --query "web app" --domain "Web Development" --top 5
  python app_cli.py --query "deep learning" --language "python"
  python app_cli.py --similar 42
        """
    )
    
    # Query options
    parser.add_argument("--query", "-q", type=str, help="Search query text")
    parser.add_argument("--similar", "-s", type=int, help="Find projects similar to this project ID")
    
    # Filter options
    parser.add_argument("--domain", "-d", type=str, help="Filter by domain (partial match)")
    parser.add_argument("--language", "-l", type=str, help="Filter by programming language")
    parser.add_argument("--difficulty", type=str, help="Filter by difficulty level")
    
    # Output options
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    # Info options
    parser.add_argument("--list-domains", action="store_true", help="List all domains")
    parser.add_argument("--list-languages", action="store_true", help="List all languages")
    parser.add_argument("--list-difficulties", action="store_true", help="List all difficulty levels")
    
    args = parser.parse_args()
    
    # Initialize recommender
    try:
        rec = ProjectRecommender()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun 'python build_pipeline.py' first to build artifacts.")
        return
    
    # Handle info options
    if args.list_domains:
        print("\nAvailable Domains:")
        for d in rec.get_domains():
            print(f"  - {d}")
        return
    
    if args.list_languages:
        print("\nAvailable Languages:")
        for l in rec.get_languages():
            print(f"  - {l}")
        return
    
    if args.list_difficulties:
        print("\nDifficulty Levels:")
        for d in rec.get_difficulty_levels():
            print(f"  - {d}")
        return
    
    # Handle similar project search
    if args.similar:
        try:
            results = rec.recommend_by_project_id(args.similar, top_n=args.top)
            print(f"\nProjects similar to ID {args.similar}:\n")
        except ValueError as e:
            print(f"Error: {e}")
            return
    elif args.query:
        results = rec.recommend(
            query=args.query,
            top_n=args.top,
            domain_filter=args.domain,
            language_filter=args.language,
            difficulty_filter=args.difficulty,
            min_score=args.min_score
        )
        print(f"\nResults for: '{args.query}'")
        if args.domain:
            print(f"  Domain filter: {args.domain}")
        if args.language:
            print(f"  Language filter: {args.language}")
        if args.difficulty:
            print(f"  Difficulty filter: {args.difficulty}")
        print()
    else:
        parser.print_help()
        return
    
    if not results:
        print("No results found.")
        return
    
    # Format output
    if args.verbose:
        for i, r in enumerate(results, 1):
            print(f"{'='*60}")
            print(f"#{i} - {r['title']} (Score: {r['score']})")
            print(f"{'='*60}")
            print(f"  Project ID:  {r['project_id']}")
            print(f"  Domain:      {r['domain']}")
            print(f"  Language:    {r['language']}")
            print(f"  Difficulty:  {r['difficulty']}")
            print(f"  Build Time:  {r['build_time']}")
            print(f"  Keywords:    {r['keywords']}")
            print(f"  Libraries:   {r['libraries']}")
            print(f"  Description: {r['description'][:200]}...")
            print()
    else:
        # Table format
        table_data = []
        for r in results:
            table_data.append([
                r['project_id'],
                r['title'][:40] + "..." if len(r['title']) > 40 else r['title'],
                r['domain'][:20] if r['domain'] else "",
                r['language'][:15] if r['language'] else "",
                r['score']
            ])
        
        print(tabulate(
            table_data,
            headers=["ID", "Title", "Domain", "Language", "Score"],
            tablefmt="simple"
        ))
    
    print(f"\nFound {len(results)} results.")


if __name__ == "__main__":
    main()
