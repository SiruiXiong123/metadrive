import importlib,sys
sys.path.insert(0, r'C:\Users\37945\OneDrive\Desktop\metadrive\metadrive')
try:
    import metadrive.obs.top_down_obs as m
    print('Imported OK:', m.TopDownObservation)
except Exception as e:
    print('Import failed:', type(e), e)
    import traceback; traceback.print_exc()
