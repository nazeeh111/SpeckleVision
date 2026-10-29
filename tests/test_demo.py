"""Actual CPU U-net on bundled images with temporary random weights, not quality."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
import demo
from utils.network import Unet

ROOT=Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory(prefix='specklevision-random-contract-')
        cls.folder=Path(cls.temp.name)
        torch.manual_seed(9)
        cls.model=Unet().eval()
        cls.checkpoint=cls.folder/'random-state-dict.pth'
        torch.save(cls.model.state_dict(),cls.checkpoint)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_rgb_resize_channel_and_crop_contract_matches_independent_pil(self):
        image=next((ROOT/'datasets/SubsampledData/test').glob('*.png'))
        with Image.open(image) as source:
            expected=np.asarray(source.convert('RGB').resize((256,128),Image.Resampling.BICUBIC))[:,:128,0]
        np.testing.assert_array_equal(demo.prepare_image(image).numpy()[0,0],(expected.astype(np.float32)/255-.5)/.5)

    def test_real_cpu_checkpoint_run_and_changed_image(self):
        paths=sorted((ROOT/'datasets/SubsampledData/test').glob('*.png'))[:2]
        collection=self.folder/'two-recorded-correlations'
        collection.mkdir()
        for p in paths:(collection/p.name).write_bytes(p.read_bytes())
        output=self.folder/'new-result'
        report=demo.run(self.checkpoint,collection,output,limit=2)
        self.assertEqual(len(report['inputs']),2)
        outputs=[]
        with torch.inference_mode():
            for i, image in enumerate(paths):
                expected=self.model(demo.prepare_image(image))[0,0,32:96,32:96].numpy()
                actual=np.load(output/f'{i}_recon.npy')
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(actual.shape,(64,64))
                outputs.append(actual)
        self.assertGreater(np.max(abs(outputs[0]-outputs[1])),0)
        self.assertIn('Not determined',report['training_provenance'])
        self.assertEqual(json.loads((output/'run.json').read_text())['checkpoint_sha256'],demo.digest(self.checkpoint))
        before=(output/'run.json').read_bytes()
        with self.assertRaises(FileExistsError):demo.run(self.checkpoint,collection,output)
        self.assertEqual((output/'run.json').read_bytes(),before)
        # Dropout and BatchNorm must stay in eval mode for repeatable inference.
        loaded=demo.load_model(self.checkpoint,'cpu')
        self.assertFalse(loaded.training)
        with torch.inference_mode():
            a=loaded(demo.prepare_image(paths[0])); b=loaded(demo.prepare_image(paths[0]))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_missing_weights_do_not_create_output(self):
        out=self.folder/'missing-output'
        with self.assertRaises(FileNotFoundError):demo.run(self.folder/'missing.pth',ROOT/'datasets',out)
        self.assertFalse(out.exists())

    def test_legacy_whole_model_requires_explicit_trust(self):
        legacy=self.folder/'legacy-random.pth'
        # Small width verifies compatibility with the historical serialized class.
        torch.save(Unet(ngf=4),legacy)
        with self.assertRaisesRegex(ValueError,'trusted legacy'):demo.load_model(legacy,'cpu')
        model=demo.load_model(legacy,'cpu',trust_legacy_pickle=True)
        self.assertIsInstance(model,Unet)
        self.assertFalse(model.training)


if __name__=='__main__':unittest.main()
