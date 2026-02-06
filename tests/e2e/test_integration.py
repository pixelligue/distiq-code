"""
End-to-End Testing for Distiq Code

Tests the complete flow:
1. Code Indexing
2. Semantic Search
3. Web Search
4. Orchestrator
"""

import asyncio
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv()


def test_header(name: str):
    """Print test header."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")


def test_pass(msg: str):
    print(f"  ✅ {msg}")


def test_fail(msg: str):
    print(f"  ❌ {msg}")


async def test_indexing():
    """Test code indexing."""
    test_header("1. Code Indexing")
    
    try:
        from distiq_code.indexing import get_indexer
        
        # Index current project
        indexer = get_indexer(Path("."))
        
        # Check if already indexed
        if indexer.vector_store and indexer.vector_store.size > 0:
            test_pass(f"Index exists: {indexer.vector_store.size} chunks")
        else:
            print("  Indexing project (first time)...")
            stats = indexer.index(show_progress=True)
            test_pass(f"Indexed {stats.get('total_chunks', 0)} chunks")
        
        return True
        
    except Exception as e:
        test_fail(f"Indexing failed: {e}")
        return False


async def test_semantic_search():
    """Test semantic code search."""
    test_header("2. Semantic Code Search")
    
    try:
        from distiq_code.indexing import get_indexer
        
        indexer = get_indexer(Path("."))
        
        # Search for something
        query = "async function cost tracking"
        results = indexer.search(query, k=3)
        
        if results:
            test_pass(f"Found {len(results)} results for '{query}'")
            for r in results[:2]:
                print(f"      - {r['file_path']}:{r['start_line']}")
            return True
        else:
            test_fail("No results found")
            return False
            
    except Exception as e:
        test_fail(f"Search failed: {e}")
        return False


async def test_web_search():
    """Test web search."""
    test_header("3. Web Search (Jina AI)")
    
    try:
        from distiq_code.tools.web_search import smart_search, read_url
        
        # Test search
        query = "Python dataclass frozen"
        results = await smart_search(query, 2)
        
        if results.results:
            test_pass(f"Search returned {len(results.results)} results")
            for r in results.results[:2]:
                print(f"      - {r.title[:50]}...")
        else:
            test_fail("No search results")
            return False
        
        # Test URL reader
        content = await read_url("https://httpbin.org/html")
        if len(content) > 100:
            test_pass(f"URL reader works ({len(content)} chars)")
        else:
            test_fail("URL reader returned empty content")
            return False
        
        return True
        
    except Exception as e:
        test_fail(f"Web search failed: {e}")
        return False


async def test_tools():
    """Test tool registry."""
    test_header("4. Tools Registry")
    
    try:
        from distiq_code.tools import get_tool_registry
        
        registry = get_tool_registry()
        tools = registry.list_tools()
        
        expected = ["Read", "Write", "Glob", "Grep", "Bash", "Search", "WebSearch", "ReadURL"]
        found = [t.name for t in tools]
        
        missing = set(expected) - set(found)
        if not missing:
            test_pass(f"All {len(tools)} tools registered")
            print(f"      Tools: {', '.join(found)}")
            return True
        else:
            test_fail(f"Missing tools: {missing}")
            return False
            
    except Exception as e:
        test_fail(f"Tools test failed: {e}")
        return False


async def test_skills():
    """Test skills registry."""
    test_header("5. Skills Registry")
    
    try:
        from distiq_code.skills import list_skills, get_skill
        
        skills = list_skills()
        
        if len(skills) >= 5:
            test_pass(f"Found {len(skills)} skills")
            print(f"      Skills: {', '.join(skills)}")
            
            # Test getting a skill
            skill = get_skill("refactor")
            if skill:
                test_pass(f"Skill 'refactor' loaded")
            
            return True
        else:
            test_fail(f"Only {len(skills)} skills found")
            return False
            
    except Exception as e:
        test_fail(f"Skills test failed: {e}")
        return False


async def test_providers():
    """Test provider registry."""
    test_header("6. Providers")
    
    try:
        from distiq_code.providers import get_available_providers
        
        providers = get_available_providers()
        
        available = [p["name"] for p in providers if p["available"]]
        
        if available:
            test_pass(f"Available providers: {len(available)}")
            for p in providers:
                status = "✓" if p["available"] else "✗"
                print(f"      {status} {p['name']}")
            return True
        else:
            test_fail("No providers available")
            return False
            
    except Exception as e:
        test_fail(f"Providers test failed: {e}")
        return False


async def test_orchestrator():
    """Test orchestrator (without actual LLM call)."""
    test_header("7. Orchestrator")
    
    try:
        from distiq_code.orchestrator import Orchestrator, OrchestratorConfig
        
        # Create orchestrator
        config = OrchestratorConfig(
            auto_index=False,  # Don't index in test
            enable_cache=True,
        )
        orchestrator = Orchestrator(config=config)
        
        test_pass("Orchestrator created")
        
        # Test context building
        context = await orchestrator._build_context("find async functions")
        if context:
            test_pass(f"Context builder works ({len(context.chunks)} chunks)")
        else:
            test_pass("Context builder works (no index)")
        
        # Test complexity classification
        complexity = await orchestrator._classify_complexity("add a docstring")
        test_pass(f"Classifier works: 'add a docstring' → {complexity}")
        
        complexity = await orchestrator._classify_complexity("refactor the entire auth system")
        test_pass(f"Classifier works: 'refactor...' → {complexity}")
        
        await orchestrator.close()
        return True
        
    except Exception as e:
        test_fail(f"Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "🚀 DISTIQ-CODE E2E TESTS".center(60))
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results["indexing"] = await test_indexing()
    results["semantic_search"] = await test_semantic_search()
    results["web_search"] = await test_web_search()
    results["tools"] = await test_tools()
    results["skills"] = await test_skills()
    results["providers"] = await test_providers()
    results["orchestrator"] = await test_orchestrator()
    
    # Summary
    test_header("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠️ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
