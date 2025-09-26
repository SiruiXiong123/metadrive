import sys
sys.path.insert(0, r'C:\Users\37945\OneDrive\Desktop\metadrive\metadrive')
try:
    from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
    from metadrive.envs.top_down_env import TopDownMetaDriveEnvV2
    print('TopDownMultiChannel.RESOLUTION =', TopDownMultiChannel.RESOLUTION)
    print('TopDownMetaDriveEnvV2.default resolution_size =', TopDownMetaDriveEnvV2.default_config().get('resolution_size'))
    print('OK')
except Exception as e:
    print('Import failed:', type(e), e)
    import traceback; traceback.print_exc()
