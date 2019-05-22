import numpy as np
from tensorboardX import SummaryWriter


writer = SummaryWriter()
for epoch in range(100):
    writer.add_scalar('/scala/test',np.random.rand(),epoch)
    writer.add_scalar('scalar/scalar_tst',{'xsinx': epoch*np.sin(epoch),'xcosx': epoch*np.sin(epoch)},epoch)

writer.close()