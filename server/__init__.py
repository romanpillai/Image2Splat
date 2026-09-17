"""Img2Splat beta server package.

Split out of a single 4312-line server.py. The division is by PIPELINE STAGE,
which is also how the operator thinks about the tool:

  config           every external path, read from ../config.json
  jobs             the two lanes, the lock, the log ring buffer, error stamping
  state            project directories, state.json, the UI snapshot
  models           request bodies, shared by more than one route module
  common           resolvers both an approval preview and its request must share
  routes_project   create / list / clone / rename / export / import / upload
  routes_orbit     calibration, the lifted cloud, the control render
  routes_generate  fal: plan, send, describe, retime, approve
  routes_passes    the two extra orbits rendered from the trained splat
  routes_dataset   dataset_complete/ and the matte models
  routes_splat     COLMAP + Brush
  routes_system    health, /api/log, status, VRAM
  app              the FastAPI app itself
"""
