from glotzformats import reader
r = reader.GSDHOOMDFileReader()
with open('test.gsd', 'rb') as f:
    traj = r.read(f)
    traj.load_arrays()
    pos = traj.positions.copy()
