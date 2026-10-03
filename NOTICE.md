# Source and additions

The retained implementation comes from [ricedsp/Deep_Inverse_Correlography](https://github.com/ricedsp/Deep_Inverse_Correlography/tree/b7f720fdeab6a028415710a87634e66d6f95e154), snapshot `b7f720fdeab6a028415710a87634e66d6f95e154`. The source is associated with the Optica 2020 paper Deep-inverse correlography: towards real-time high-resolution non-line-of-sight imaging; the named authors are Christopher Metzler and the cited research collaborators.

SpeckleVision adds repository presentation, setup guidance and bounded CPU verification records, maintained by nazeeh111. These additions do not establish new authorship of the research method or independently reproduce the source-reported scientific results.

The complete original [LICENSE](LICENSE) restores Christopher Metzler’s 2020 MIT notice and the separate CycleGAN terms for Jun-Yan Zhu and Taesung Park. The original author comments in `create_training_data.m` and `utils/xcorr2.py` are restored without numerical changes. New additions use [LICENSE-branding](LICENSE-branding). BSD-500 and external checkpoints/raw measurements retain their own terms. The random-weight CPU check does not establish trained reconstruction accuracy.
