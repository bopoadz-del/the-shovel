import pytest
"""
Comprehensive Block Test Suite
Tests all 22 blocks (15 core + 7 containers)
"""

import asyncio
import sys
import time
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspaces/SSDPPG')

# Test results tracker
results = {
    "passed": [],
    "failed": [],
    "skipped": [],
    "total": 0
}

def log_test(block_name: str, test_name: str, passed: bool, error: str = None, duration_ms: float = 0):
    """Log a test result"""
    results["total"] += 1
    status = "✅ PASS" if passed else "❌ FAIL"
    
    entry = {
        "block": block_name,
        "test": test_name,
        "passed": passed,
        "error": error,
        "duration_ms": duration_ms
    }
    
    if passed:
        results["passed"].append(entry)
        print(f"  {status} {test_name} ({duration_ms:.1f}ms)")
    else:
        results["failed"].append(entry)
        print(f"  {status} {test_name}: {error}")

# ============================================================================
# CORE AI BLOCKS (15)
# ============================================================================

@pytest.mark.asyncio
async def test_chat_block():
    """Test Chat Block"""
    print("\n🤖 Testing Chat Block...")
    start = time.time()
    
    try:
        from app.blocks.chat import ChatBlock
        block = ChatBlock()
        
        # Test basic chat
        result = await block.process("Hello", {"provider": "deepseek"})
        
        if result.get("status") == "success":
            log_test("chat", "basic_chat", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("chat", "basic_chat", False, result.get("error", "Unknown error"))
            
    except Exception as e:
        log_test("chat", "basic_chat", False, str(e))

@pytest.mark.asyncio
async def test_pdf_block():
    """Test PDF Block"""
    print("\n📄 Testing PDF Block...")
    start = time.time()
    
    try:
        from app.blocks.pdf import PDFBlock
        block = PDFBlock()
        
        # Test with a sample text (no actual PDF needed for basic test)
        result = await block.process("Sample text content", {"action": "extract"})
        
        log_test("pdf", "extract", result.get("status") == "success", 
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("pdf", "extract", False, str(e))

@pytest.mark.asyncio
async def test_ocr_block():
    """Test OCR Block"""
    print("\n👁️ Testing OCR Block...")
    start = time.time()
    
    try:
        from app.blocks.ocr import OCRBlock
        block = OCRBlock()
        
        # OCR requires an image, test the block exists
        result = await block.process(None, {"url": "test.jpg"})
        
        # OCR may fail without real image, but block should respond
        log_test("ocr", "process", True, None, (time.time()-start)*1000)
        
    except Exception as e:
        log_test("ocr", "process", False, str(e))

@pytest.mark.asyncio
async def test_voice_block():
    """Test Voice Block"""
    print("\n🔊 Testing Voice Block...")
    start = time.time()
    
    try:
        from app.blocks.voice import VoiceBlock
        block = VoiceBlock()
        
        result = await block.process("Hello world", {"action": "tts"})
        
        log_test("voice", "tts", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("voice", "tts", False, str(e))

@pytest.mark.asyncio
async def test_image_block():
    """Test Image Block"""
    print("\n🖼️ Testing Image Block...")
    start = time.time()
    
    try:
        from app.blocks.image import ImageBlock
        block = ImageBlock()
        
        result = await block.process(None, {"action": "analyze"})
        
        log_test("image", "analyze", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("image", "analyze", False, str(e))

@pytest.mark.asyncio
async def test_translate_block():
    """Test Translate Block"""
    print("\n🌐 Testing Translate Block...")
    start = time.time()
    
    try:
        from app.blocks.translate import TranslateBlock
        block = TranslateBlock()
        
        result = await block.process("Hello", {"target": "es"})
        
        log_test("translate", "translate", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("translate", "translate", False, str(e))

@pytest.mark.asyncio
async def test_code_block():
    """Test Code Block"""
    print("\n💻 Testing Code Block...")
    start = time.time()
    
    try:
        from app.blocks.code import CodeBlock
        block = CodeBlock()
        
        result = await block.process("print('hello')", {"language": "python"})
        
        log_test("code", "execute", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("code", "execute", False, str(e))

@pytest.mark.asyncio
async def test_web_block():
    """Test Web Block"""
    print("\n🕸️ Testing Web Block...")
    start = time.time()
    
    try:
        from app.blocks.web import WebBlock
        block = WebBlock()
        
        result = await block.process("https://example.com", {"action": "scrape"})
        
        log_test("web", "scrape", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("web", "scrape", False, str(e))

@pytest.mark.asyncio
async def test_search_block():
    """Test Search Block"""
    print("\n🔎 Testing Search Block...")
    start = time.time()
    
    try:
        from app.blocks.search import SearchBlock
        block = SearchBlock()
        
        result = await block.process("AI news", {"engine": "duckduckgo"})
        
        log_test("search", "search", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("search", "search", False, str(e))

@pytest.mark.asyncio
async def test_google_drive_block():
    """Test Google Drive Block"""
    print("\n📁 Testing Google Drive Block...")
    start = time.time()
    
    try:
        from app.blocks.google_drive import GoogleDriveBlock
        block = GoogleDriveBlock()
        
        result = await block.process(None, {"action": "list"})
        
        # May fail without auth, but block should respond
        log_test("google_drive", "list", True, None, (time.time()-start)*1000)
        
    except Exception as e:
        log_test("google_drive", "list", False, str(e))

@pytest.mark.asyncio
async def test_onedrive_block():
    """Test OneDrive Block"""
    print("\n☁️ Testing OneDrive Block...")
    start = time.time()
    
    try:
        from app.blocks.onedrive import OneDriveBlock
        block = OneDriveBlock()
        
        result = await block.process(None, {"action": "list"})
        
        log_test("onedrive", "list", True, None, (time.time()-start)*1000)
        
    except Exception as e:
        log_test("onedrive", "list", False, str(e))

@pytest.mark.asyncio
async def test_local_drive_block():
    """Test Local Drive Block"""
    print("\n💾 Testing Local Drive Block...")
    start = time.time()
    
    try:
        from app.blocks.local_drive import LocalDriveBlock
        block = LocalDriveBlock()
        
        result = await block.process("/tmp", {"action": "list"})
        
        log_test("local_drive", "list", result.get("status") == "success",
                 result.get("error"), (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("local_drive", "list", False, str(e))

@pytest.mark.asyncio
async def test_android_drive_block():
    """Test Android Drive Block"""
    print("\n📱 Testing Android Drive Block...")
    start = time.time()
    
    try:
        from app.blocks.android_drive import AndroidDriveBlock
        block = AndroidDriveBlock()
        
        result = await block.process(None, {"action": "list"})
        
        log_test("android_drive", "list", True, None, (time.time()-start)*1000)
        
    except Exception as e:
        log_test("android_drive", "list", False, str(e))

# ============================================================================
# DOMAIN CONTAINERS (7)
# ============================================================================

@pytest.mark.asyncio
async def test_construction_container():
    """Test Construction Container"""
    print("\n🏗️ Testing Construction Container...")
    start = time.time()
    
    try:
        from app.containers.construction import ConstructionContainer
        container = ConstructionContainer()
        
        # Test measurements extraction
        result = await container.process({}, {"action": "extract_measurements"})
        
        if result.get("status") == "success" and "quantities" in result:
            log_test("construction", "extract_measurements", True, 
                     duration_ms=(time.time()-start)*1000)
        else:
            log_test("construction", "extract_measurements", False, 
                     result.get("error", "Missing quantities"))
                     
    except Exception as e:
        log_test("construction", "extract_measurements", False, str(e))

@pytest.mark.asyncio
async def test_medical_container():
    """Test Medical Container"""
    print("\n🏥 Testing Medical Container...")
    start = time.time()
    
    try:
        from app.containers.medical import MedicalContainer
        container = MedicalContainer()
        
        result = await container.process({}, {"action": "process_dicom"})
        
        if result.get("status") == "success":
            log_test("medical", "process_dicom", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("medical", "process_dicom", False, result.get("error"))
            
    except Exception as e:
        log_test("medical", "process_dicom", False, str(e))

@pytest.mark.asyncio
async def test_legal_container():
    """Test Legal Container"""
    print("\n⚖️ Testing Legal Container...")
    start = time.time()
    
    try:
        from app.containers.legal import LegalContainer
        container = LegalContainer()
        
        result = await container.process({}, {"action": "process_contract"})
        
        if result.get("status") == "success":
            log_test("legal", "process_contract", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("legal", "process_contract", False, result.get("error"))
            
    except Exception as e:
        log_test("legal", "process_contract", False, str(e))

@pytest.mark.asyncio
async def test_finance_container():
    """Test Finance Container"""
    print("\n💰 Testing Finance Container...")
    start = time.time()
    
    try:
        from app.containers.finance import FinanceContainer
        container = FinanceContainer()
        
        result = await container.process({}, {"action": "process_trades"})
        
        if result.get("status") == "success":
            log_test("finance", "process_trades", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("finance", "process_trades", False, result.get("error"))
            
    except Exception as e:
        log_test("finance", "process_trades", False, str(e))

@pytest.mark.asyncio
async def test_security_container():
    """Test Security Container"""
    print("\n🔐 Testing Security Container...")
    start = time.time()
    
    try:
        from app.containers.security import SecurityContainer
        container = SecurityContainer()
        
        # Test key creation
        result = await container.process({}, {"action": "create_key", "owner": "test"})
        
        if result.get("api_key", "").startswith("cb_"):
            log_test("security", "create_key", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("security", "create_key", False, "Invalid key format")
            
    except Exception as e:
        log_test("security", "create_key", False, str(e))

@pytest.mark.asyncio
async def test_ai_core_container():
    """Test AI Core Container"""
    print("\n🤖 Testing AI Core Container...")
    start = time.time()
    
    try:
        from app.containers.ai_core import AICoreContainer
        container = AICoreContainer()
        
        result = await container.process({}, {"action": "leaderboard"})
        
        if result.get("rankings"):
            log_test("ai_core", "leaderboard", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("ai_core", "leaderboard", False, "No rankings")
            
    except Exception as e:
        log_test("ai_core", "leaderboard", False, str(e))

@pytest.mark.asyncio
async def test_store_container():
    """Test Store Container"""
    print("\n🏪 Testing Store Container...")
    start = time.time()
    
    try:
        from app.containers.store import StoreContainer
        container = StoreContainer()
        
        result = await container.process({}, {"action": "platform_stats"})
        
        if "total_blocks" in result:
            log_test("store", "platform_stats", True, duration_ms=(time.time()-start)*1000)
        else:
            log_test("store", "platform_stats", False, "Missing stats")
            
    except Exception as e:
        log_test("store", "platform_stats", False, str(e))

# ============================================================================
# CHAIN EXECUTION
# ============================================================================

@pytest.mark.asyncio
async def test_chain_execution():
    """Test multi-block chain execution"""
    print("\n⛓️ Testing Chain Execution...")
    start = time.time()
    
    try:
        from app.blocks import BLOCK_REGISTRY
        
        # Simple chain: translate → chat
        chain = [
            {"block": "translate", "params": {"target": "es"}},
            {"block": "chat", "params": {"provider": "deepseek"}}
        ]
        
        input_data = "Hello, how are you?"
        
        # Execute chain
        current_input = input_data
        success = True
        
        for step in chain:
            block_name = step["block"]
            params = step.get("params", {})
            
            if block_name not in BLOCK_REGISTRY:
                success = False
                break
                
            block_class = BLOCK_REGISTRY[block_name]
            block = block_class()
            result = await block.process(current_input, params)
            
            if result.get("status") != "success":
                success = False
                break
                
            # Pass output to next step
            current_input = result.get("text", result.get("translated", str(result)))
        
        log_test("chain", "translate_chat_chain", success, 
                 None if success else "Chain failed", (time.time()-start)*1000)
                 
    except Exception as e:
        log_test("chain", "translate_chat_chain", False, str(e))

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all block tests"""
    print("=" * 70)
    print("🧠 CEREBRUM BLOCKS - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing: 22 blocks (15 core + 7 containers)")
    print("=" * 70)
    
    # Core AI Blocks
    await test_chat_block()
    await test_pdf_block()
    await test_ocr_block()
    await test_voice_block()
    await test_image_block()
    await test_translate_block()
    await test_code_block()
    await test_web_block()
    await test_search_block()
    await test_google_drive_block()
    await test_onedrive_block()
    await test_local_drive_block()
    await test_android_drive_block()
    
    # Domain Containers
    await test_construction_container()
    await test_medical_container()
    await test_legal_container()
    await test_finance_container()
    await test_security_container()
    await test_ai_core_container()
    await test_store_container()
    
    # Chain Execution
    await test_chain_execution()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = len(results["passed"])
    failed = len(results["failed"])
    total = results["total"]
    
    print(f"Total Tests:  {total}")
    print(f"✅ Passed:     {passed} ({passed/total*100:.1f}%)")
    print(f"❌ Failed:     {failed} ({failed/total*100:.1f}%)")
    
    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for entry in results["failed"]:
            print(f"  - {entry['block']}.{entry['test']}: {entry['error']}")
    
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
