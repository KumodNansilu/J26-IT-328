import ast
import sys

filepath = "c:/Users/User/Desktop/J26-IT-328/MindSync-Enterprise/app/ui/pages/video_analysis_page.py"

try:
    with open(filepath) as f:
        content = f.read()
    ast.parse(content)
    print("✅ Syntax OK - file parses correctly")
    
    # Check for the specific VBSSpecMissingError message
    if "Not available — scoring method requires approval" in content:
        print("✅ VBSSpecMissingError message found: 'Not available — scoring method requires approval'")
    else:
        print("❌ VBSSpecMissingError message NOT found")
    
    # Check for the old message
    if "No approved VBS formula" in content:
        print("⚠️  Old message still present: 'No approved VBS formula...'")
    else:
        print("✅ Old message replaced correctly")
    
    # Check AnalysisResult dataclass has vbs fields
    if "vbs_available: bool = False" in content:
        print("✅ vbs_available field found in AnalysisResult")
    else:
        print("❌ vbs_available field NOT found")
    
    if "vbs_message: str = \"\"" in content:
        print("✅ vbs_message field found in AnalysisResult")
    else:
        print("❌ vbs_message field NOT found")
    
    # Check _on_done slot has lbl_vbs.setText
    if "self.lbl_vbs.setText" in content:
        print("✅ lbl_vbs.setText found in _on_done slot")
    else:
        print("❌ lbl_vbs.setText NOT found")
    
    # Check disclaimer
    if "Emotion predictions are experimental facial-expression outputs" in content:
        print("✅ Experimental disclaimer found")
    else:
        print("❌ Experimental disclaimer NOT found")
    
except FileNotFoundError:
    print(f"❌ File NOT found: {filepath}")
    sys.exit(1)
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

print("\n--- Check Complete ---")