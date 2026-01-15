#!/usr/bin/env python3
"""Quick test for agriculture keyword"""

from recommender import ProjectRecommender

print("Loading recommender...")
rec = ProjectRecommender()

print("\n" + "="*80)
print("TESTING: agriculture")
print("="*80)

results = rec.recommend('agriculture', top_n=10, min_score=0.0)

print(f"\nFound {len(results)} results:")

for i, r in enumerate(results[:10], 1):
    print(f"\n{i}. {r['title']}")
    print(f"   Domain: {r['domain']}")
    print(f"   Score: {r['score']:.4f}")
    if 'agriculture' in r['title'].lower() or 'agriculture' in r['description'].lower():
        print(f"   ✓ Contains 'agriculture'")

print("\n" + "="*80)
print("TESTING: machine learning")
print("="*80)

results = rec.recommend('machine learning', top_n=5, min_score=0.0)

print(f"\nFound {len(results)} results:")

for i, r in enumerate(results[:5], 1):
    print(f"{i}. {r['title']} (Score: {r['score']:.4f})")

print("\n✓ Test complete!")
