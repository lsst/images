# This file is part of lsst-images.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

import os
import unittest

import astropy.io.fits
import astropy.units as u
import numpy as np
from astro_metadata_translator import ObservationInfo

import lsst.utils.tests
from lsst.images import Box, DetectorFrame, Image, Projection, SkyFrame, Transform
from lsst.images.tests import assert_close, assert_images_equal, compare_image_to_legacy

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)

# Very simple AST mapping that can be used for serialization as a Projection.
TEST_FRAMESET = """
Begin FrameSet
#   Title = "ICRS coordinates; gnomonic projection"
#   Naxes = 2
#   Domain = "SKY"
#   Epoch = 2000
#   Lbl1 = "Right ascension"
#   Lbl2 = "Declination"
#   System = "ICRS"
#   Uni1 = "hh:mm:ss.s"
#   Uni2 = "ddd:mm:ss"
#   Dir1 = 0
#   Bot2 = -1.5707963267948966
#   Top2 = 1.5707963267948966
 IsA Frame
    Nframe = 5
#   Base = 1
    Currnt = 5
    Lnk2 = 1
    Lnk3 = 2
    Lnk4 = 3
    Lnk5 = 4
    Frm1 =
       Begin Frame
#         Title = "2-d coordinate system"
          Naxes = 2
          Domain = "PIXELS"
#         Lbl1 = "Axis 1"
#         Lbl2 = "Axis 2"
          Ax1 =
             Begin Axis
             End Axis
          Ax2 =
             Begin Axis
             End Axis
       End Frame
    Frm2 =
       Begin Frame
#         Title = "2-d coordinate system"
          Naxes = 2
          Domain = "NORMEDPIXELS"
#         Lbl1 = "Axis 1"
#         Lbl2 = "Axis 2"
          Ax1 =
             Begin Axis
             End Axis
          Ax2 =
             Begin Axis
             End Axis
       End Frame
    Frm3 =
       Begin Frame
#         Title = "2-d coordinate system"
          Naxes = 2
          Domain = "INTERMEDIATE0"
#         Lbl1 = "Axis 1"
#         Lbl2 = "Axis 2"
          Ax1 =
             Begin Axis
             End Axis
          Ax2 =
             Begin Axis
             End Axis
       End Frame
    Frm4 =
       Begin Frame
#         Title = "2-d coordinate system"
          Naxes = 2
          Domain = "IWC"
#         Lbl1 = "Axis 1"
#         Lbl2 = "Axis 2"
          Ax1 =
             Begin Axis
             End Axis
          Ax2 =
             Begin Axis
             End Axis
       End Frame
    Frm5 =
       Begin SkyFrame
          Ident = " "
       IsA Object
#         Title = "ICRS coordinates; gnomonic projection"
          Naxes = 2
#         Domain = "SKY"
#         Epoch = 2000
#         Lbl1 = "Right ascension"
#         Lbl2 = "Declination"
          System = "ICRS"
          AlSys = "ICRS"
#         Uni1 = "hh:mm:ss.s"
#         Uni2 = "ddd:mm:ss"
#         Dir1 = 0
#         Bot2 = -1.5707963267948966
#         Top2 = 1.5707963267948966
          Ax1 =
             Begin SkyAxis
             End SkyAxis
          Ax2 =
             Begin SkyAxis
             End SkyAxis
       IsA Frame
          Proj = "gnomonic"
#         SkyTol = 0.001
          SRefIs = "Ignored"
          SRef1 = 0.93073584964951706
          SRef2 = -0.48994978249425891
       End SkyFrame
    Map2 =
       Begin PolyMap
          Nin = 2
       IsA Mapping
          MPF1 = 1
          MPF2 = 1
          NCF1 = 3
          NCF2 = 3
          CF1 = -1
          CF2 = 0.00049127978383689509
          CF3 = 0
          CF4 = -1
          CF5 = 0
          CF6 = 0.00050012503125781451
          PF3 = 1
          PF6 = 1
          PF9 = 1
          PF12 = 1
          IterInv = 1
          NiterInv = 10
          TolInv = 9.9999999999999995e-08
       End PolyMap
    Map3 =
       Begin PolyMap
          Nin = 2
       IsA Mapping
          MPF1 = 5
          MPF2 = 5
          NCF1 = 21
          NCF2 = 21
          CF1 = -0.28442541183584946
          CF2 = 0.025196410450119774
          CF3 = 0.10850740257872329
          CF4 = -1.8117214171184971e-05
          CF5 = -2.5411973475688556e-05
          CF6 = -2.7055625077518308e-05
          CF7 = 5.9561505359004041e-06
          CF8 = 4.3469327876298142e-06
          CF9 = -1.9295976474719754e-06
          CF10 = 8.9276781617670002e-06
          CF11 = -3.0253037572317978e-06
          CF12 = -3.1691916359221918e-06
          CF13 = -2.5343291201044216e-06
          CF14 = 2.5103885008176705e-06
          CF15 = -3.3353885943820633e-06
          CF16 = 7.0231915037332632e-07
          CF17 = 4.433637931327871e-06
          CF18 = 1.9014913722005829e-06
          CF19 = 1.4071986461672182e-08
          CF20 = 4.0071804654658158e-07
          CF21 = 1.2864360459629361e-07
          CF22 = -0.11865465113108672
          CF23 = 0.11043904923265806
          CF24 = -0.024679579661590779
          CF25 = -2.7450540092086099e-05
          CF26 = -1.7393341912958501e-05
          CF27 = -3.6787946682862772e-06
          CF28 = 1.0470311024552662e-05
          CF29 = -7.0809073268373886e-07
          CF30 = 1.5779574342544132e-07
          CF31 = 5.7811785237108776e-07
          CF32 = -4.2622855209830129e-06
          CF33 = -2.1386278247017794e-06
          CF34 = -2.9383420175814586e-06
          CF35 = -2.5116133254473027e-07
          CF36 = -1.3619505424895891e-06
          CF37 = 1.3034570560858815e-06
          CF38 = 3.6229356255468614e-06
          CF39 = 6.205160891544347e-07
          CF40 = 1.3650773593104822e-06
          CF41 = 3.5979635479701857e-06
          CF42 = 2.1174941282631594e-07
          PF3 = 1
          PF6 = 1
          PF7 = 2
          PF9 = 1
          PF10 = 1
          PF12 = 2
          PF13 = 3
          PF15 = 2
          PF16 = 1
          PF17 = 1
          PF18 = 2
          PF20 = 3
          PF21 = 4
          PF23 = 3
          PF24 = 1
          PF25 = 2
          PF26 = 2
          PF27 = 1
          PF28 = 3
          PF30 = 4
          PF31 = 5
          PF33 = 4
          PF34 = 1
          PF35 = 3
          PF36 = 2
          PF37 = 2
          PF38 = 3
          PF39 = 1
          PF40 = 4
          PF42 = 5
          PF45 = 1
          PF48 = 1
          PF49 = 2
          PF51 = 1
          PF52 = 1
          PF54 = 2
          PF55 = 3
          PF57 = 2
          PF58 = 1
          PF59 = 1
          PF60 = 2
          PF62 = 3
          PF63 = 4
          PF65 = 3
          PF66 = 1
          PF67 = 2
          PF68 = 2
          PF69 = 1
          PF70 = 3
          PF72 = 4
          PF73 = 5
          PF75 = 4
          PF76 = 1
          PF77 = 3
          PF78 = 2
          PF79 = 2
          PF80 = 3
          PF81 = 1
          PF82 = 4
          PF84 = 5
          IterInv = 1
          NiterInv = 10
          TolInv = 9.9999999999999995e-08
       End PolyMap
    Map4 =
       Begin UnitMap
          Nin = 2
       IsA Mapping
       End UnitMap
    Map5 =
       Begin CmpMap
          Nin = 2
       IsA Mapping
          MapA =
             Begin UnitMap
                Nin = 2
                IsSimp = 1
             IsA Mapping
             End UnitMap
          MapB =
             Begin CmpMap
                Nin = 2
             IsA Mapping
                InvA = 1
                MapA =
                   Begin CmpMap
                      Nin = 2
                      Invert = 1
                   IsA Mapping
                      InvA = 1
                      MapA =
                         Begin SphMap
                            Nin = 3
                            Nout = 2
                         IsA Mapping
                            UntRd = 1
                            PlrLg = 0.93073584964951761
                         End SphMap
                      MapB =
                         Begin CmpMap
                            Nin = 3
                            Nout = 2
                         IsA Mapping
                            InvA = 1
                            MapA =
                               Begin MatrixMap
                                  Nin = 3
                               IsA Mapping
                                  M0 = -0.28105200045586293
                                  M1 = -0.80205963974196071
                                  M2 = 0.52698207496721727
                                  M3 = -0.3774344917398868
                                  M4 = 0.59724394873200359
                                  M5 = 0.70770252942041767
                                  M6 = -0.88235649115582016
                                  M7 = 0
                                  M8 = -0.47058157902237213
                                  IM0 = -0.28105200045586293
                                  IM1 = -0.3774344917398868
                                  IM2 = -0.88235649115582016
                                  IM3 = -0.80205963974196071
                                  IM4 = 0.59724394873200359
                                  IM5 = 0
                                  IM6 = 0.52698207496721727
                                  IM7 = 0.70770252942041767
                                  IM8 = -0.47058157902237213
                                  Form = "Full"
                               End MatrixMap
                            MapB =
                               Begin CmpMap
                                  Nin = 3
                                  Nout = 2
                               IsA Mapping
                                  MapA =
                                     Begin SphMap
                                        Nin = 3
                                        Nout = 2
                                        Invert = 1
                                     IsA Mapping
                                        UntRd = 1
                                        PlrLg = 0
                                     End SphMap
                                  MapB =
                                     Begin CmpMap
                                        Nin = 2
                                     IsA Mapping
                                        MapA =
                                           Begin WcsMap
                                              Nin = 2
                                              Invert = 1
                                           IsA Mapping
                                              Type = "TAN"
                                           End WcsMap
                                        MapB =
                                           Begin ZoomMap
                                              Nin = 2
                                           IsA Mapping
                                              Zoom = 57.295779513082323
                                           End ZoomMap
                                     End CmpMap
                               End CmpMap
                         End CmpMap
                   End CmpMap
                MapB =
                   Begin UnitMap
                      Nin = 2
                      IsSimp = 1
                   IsA Mapping
                   End UnitMap
             End CmpMap
       End CmpMap
 End FrameSet
"""


class ImageTestCase(unittest.TestCase):
    """Tests for the Image class."""

    def test_basics(self):
        """Test basic constructor patterns."""
        image = Image(42, shape=(5, 5), metadata={"three": 3})
        assert_close(self, image.array, np.zeros([5, 5], dtype=np.int64) + 42)
        self.assertEqual(image.metadata["three"], 3)

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        image = Image(data)
        subset = image[Box.factory[:3, 1:3]]
        subset2 = image.absolute[:3, 1:3]
        assert_images_equal(self, subset2, subset, expect_view=True)

        assert_images_equal(self, image.copy(), image, expect_view=False)

        # Add an explicit bounding box and then slice it.
        image = Image(data, bbox=Box.factory[-2:1, 10:14])
        with self.assertRaises(IndexError):
            # Same slice no longer works in absolute slicing because we have
            # moved origin.
            image.absolute[:3, 1:3]
        # That slice does still work in local coordinates.
        assert_close(self, image.local[:3, 1:3].array, subset2.array)
        # And we can write an equivalent slice in absolute coordinates.
        assert_close(self, image.absolute[:0, 11:13].array, np.array([[2, 3], [6, 7]]))

        # Test __eq__ behavior.
        self.assertEqual(image[...], image)
        self.assertEqual(image.__eq__(data), NotImplemented)
        self.assertNotEqual(image, list(data))

        with self.assertRaises(ValueError):
            # bbox does not match array shape.
            Image(np.array([[1, 2, 3], [4, 5, 6]]), bbox=Box.factory[0:2, 0:4])

        with self.assertRaises(ValueError):
            # shape does not match array shape.
            Image(np.array([[2, 3, 4], [6, 7, 8]]), shape=[5, 2])

        with self.assertRaises(TypeError):
            # shape and bbox both None.
            Image()

        with self.assertRaises(ValueError):
            # Shape mismatch.
            Image(shape=[3, 6], bbox=Box.factory[-5:10, 0:10])

    def test_quantity(self):
        """Test quantities."""
        data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        data2 = data.copy() * 2.0
        image = Image(data, unit=u.mJy, bbox=Box.factory[-2:1, 3:7])

        q = image.quantity
        self.assertEqual(q[1, 0], 5.0 * u.mJy)
        image.quantity = image.array * 10.0 * u.uJy
        q = image.quantity
        self.assertEqual(q[1, 0], 0.05 * u.mJy)

        image2 = Image(data2, unit=u.Jy)
        image[Box.factory[-1:0, 5:7]] = image2.local[1:2, 2:4]
        assert_close(
            self,
            image.array,
            np.array([[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 14000.0, 16000.0], [0.09, 0.1, 0.11, 0.12]]),
        )

    def test_read_write(self):
        """Round trip through file."""
        data = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        md = {"int": 1, "float": 42.0, "bool": False, "long string header": "This is a string"}
        obsinfo = ObservationInfo(telescope="Simonyi", instrument="LSSTCam", relative_humidity=23.5)

        # Make a Projection from the AST CmpMap.
        try:
            import astshim
        except ImportError:
            projection = None
        else:
            ast_stream = astshim.StringStream(TEST_FRAMESET)
            ast_chan = astshim.Channel(ast_stream)
            ast_frame_set = ast_chan.read()
            det_frame = DetectorFrame(
                instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096]
            )
            transform = Transform(det_frame, SkyFrame.ICRS, ast_frame_set.getMapping())
            projection = Projection(transform)

        image = Image(
            data,
            unit=u.nJy,
            metadata=md,
            obs_info=obsinfo,
            bbox=Box.factory[-2:1, 3:7],
            projection=projection,
        )

        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            image.write_fits(tmpFile)

            new = Image.read_fits(tmpFile)
            self.assertEqual(new, image)

            # __eq__ does not test all components.
            self.assertEqual(new.obs_info, image.obs_info)
            self.assertEqual(new.metadata, image.metadata)
            self.maxDiff = None
            # There is no Projection __eq__ but we can check that the AST
            # native representation simplifies to the same thing.
            self.assertEqual(new.projection.show(simplified=True), image.projection.show(simplified=True))

            # Read subset.
            subset = Image.read_fits(tmpFile, bbox=Box.factory[-2:0, 5:7])
            self.assertEqual(subset, image.absolute[-2:0, 5:7])
            self.assertEqual(subset, image.local[0:2, 2:4])
            self.assertEqual(str(subset), "Image([y=-2:0, x=5:7], float64)")
            self.assertEqual(
                repr(subset),
                "Image(..., bbox=Box(y=Interval(start=-2, stop=0), x=Interval(start=5, stop=7)), "
                "dtype=dtype('float64'))",
            )

            # Check that WCS headers were written out.
            with astropy.io.fits.open(tmpFile) as hdul:
                hdu1 = hdul[1]
                hdr1 = hdu1.header
                self.assertEqual(hdr1["CTYPE1"], "RA---TAN-SIP")

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test Image.read_legacy, Image.to_legacy, and Image.from_legacy."""
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        image = Image.read_legacy(filename, preserve_quantization=True)
        try:
            from lsst.afw.image import MaskedImageFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = MaskedImageFitsReader(filename)
        legacy_image = reader.readImage()
        compare_image_to_legacy(self, image, legacy_image, expect_view=False)
        # Converting back to afw will not share memory, because
        # preserve_quantization=True makes the array read-only and to_legacy
        # has to copy in that case.
        compare_image_to_legacy(self, image, image.to_legacy(), expect_view=False)
        # Converting from afw will always share memory.
        image_view = Image.from_legacy(legacy_image)
        compare_image_to_legacy(self, image_view, legacy_image, expect_view=True)
        # Converting back to afw from the in-memory view will be another view.
        compare_image_to_legacy(self, image_view, image_view.to_legacy(), expect_view=True)


if __name__ == "__main__":
    unittest.main()
