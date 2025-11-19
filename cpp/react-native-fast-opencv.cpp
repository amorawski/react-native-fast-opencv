#include <iostream>
#include <cmath>

#include "react-native-fast-opencv.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <FOCV_Ids.hpp>
#include <FOCV_Storage.hpp>
#include <FOCV_Function.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include "FOCV_Object.hpp"
#include "ConvertImage.hpp"
#include "FOCV_JsiObject.hpp"
#include "jsi/jsi.h"
#include "opencv2/opencv.hpp"

using namespace mrousavy;

void OpenCVPlugin::installOpenCV(jsi::Runtime& runtime, std::shared_ptr<react::CallInvoker> callInvoker) {

    auto func = [=](jsi::Runtime& runtime,
                        const jsi::Value& thisArg,
                        const jsi::Value* args,
                        size_t count) -> jsi::Value {
        auto plugin = std::make_shared<OpenCVPlugin>(callInvoker);
        auto result = jsi::Object::createFromHostObject(runtime, plugin);

        return result;
    };

    auto jsiFunc = jsi::Function::createFromHostFunction(runtime,
        jsi::PropNameID::forUtf8(runtime, "__loadOpenCV"),
        1,
        func);

    runtime.global().setProperty(runtime, "__loadOpenCV", jsiFunc);

}

OpenCVPlugin::OpenCVPlugin(std::shared_ptr<react::CallInvoker> callInvoker) : _callInvoker(callInvoker) {}

jsi::Value OpenCVPlugin::get(jsi::Runtime& runtime, const jsi::PropNameID& propNameId) {
  auto propName = propNameId.utf8(runtime);

  if (propName == "frameBufferToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "frameBufferToMat"), 5,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {
        auto rows = arguments[0].asNumber();
        auto cols = arguments[1].asNumber();
        auto channels = arguments[2].asNumber();
        auto input = arguments[3].asObject(runtime);

        auto type = -1;
        if (channels == 1) {
          type = CV_8U;
        }
        if (channels == 3) {
          type = CV_8UC3;
        }
        if (channels == 4) {
          type = CV_8UC4;
        }

        if (channels == -1) {
          throw std::runtime_error("Fast OpenCV Error: Invalid channel count passed to frameBufferToMat!");
        }

        auto inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector(runtime);

        cv::Mat mat(rows, cols, type);
        memcpy(mat.data, vec.data(), (int)rows * (int)cols * (int)channels);
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  } else  if (propName == "bufferToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "bufferToMat"), 5,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {

        auto type = arguments[0].asString(runtime).utf8(runtime);
        auto rows = arguments[1].asNumber();
        auto cols = arguments[2].asNumber();
        auto channels = arguments[3].asNumber();
        auto input = arguments[4].asObject(runtime);

        auto modeType = -1;
        auto typeSize = 1;

        if(type == "uint8") {
          typeSize = 1;
          if (channels == 1) modeType = CV_8U;
          if (channels == 3) modeType = CV_8UC3;
          if (channels == 4) modeType = CV_8UC4;
        } else if(type == "uint16") {
          typeSize = 2;
          if (channels == 1) modeType = CV_16U;
          if (channels == 3) modeType = CV_16UC3;
          if (channels == 4) modeType = CV_16UC4;
        } else if(type == "int8") {
          typeSize = 1;
          if (channels == 1) modeType = CV_8S;
          if (channels == 3) modeType = CV_8SC3;
          if (channels == 4) modeType = CV_8SC4;
        } else if(type == "int16") {
          typeSize = 2;
          if (channels == 1) modeType = CV_16S;
          if (channels == 3) modeType = CV_16SC3;
          if (channels == 4) modeType = CV_16SC4;
        } else if(type == "int32") {
          typeSize = 4;
          if (channels == 1) modeType = CV_32S;
          if (channels == 3) modeType = CV_32SC3;
          if (channels == 4) modeType = CV_32SC4;
        } else if(type == "float32") {
          typeSize = 4;
          if (channels == 1) modeType = CV_32F;
          if (channels == 3) modeType = CV_32FC3;
          if (channels == 4) modeType = CV_32FC4;
        } else if(type == "float64") {
          typeSize = 8;
          if (channels == 1) modeType = CV_64F;
          if (channels == 3) modeType = CV_64FC3;
          if (channels == 4) modeType = CV_64FC4;
        }

        if (channels == -1) {
          throw std::runtime_error("Fast OpenCV Error: Invalid channel count passed to frameBufferToMat!");
        }

        auto inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector(runtime);

        cv::Mat mat(rows, cols, modeType);
        memcpy(mat.data, vec.data(), (int)rows * (int)cols * (int)channels * typeSize);
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  }
  else if (propName == "base64ToMat") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "base64ToMat"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          auto base64 = arguments[0].asString(runtime).utf8(runtime);

          auto mat = ImageConverter::str2mat(base64);
          auto id = FOCV_Storage::save(mat);

          return FOCV_JsiObject::wrap(runtime, "mat", id);
      });
    }
  else if (propName == "matToBuffer") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "matToBuffer"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

                auto id = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
                auto mat = *FOCV_Storage::get<cv::Mat>(id);

                jsi::Object value(runtime);

                value.setProperty(runtime, "cols", jsi::Value(mat.cols));
                value.setProperty(runtime, "rows", jsi::Value(mat.rows));
                value.setProperty(runtime, "channels", jsi::Value(mat.channels()));

                auto type = arguments[1].asString(runtime).utf8(runtime);
                auto size = mat.cols * mat.rows * mat.channels();

                if(type == "uint8") {
                  auto arr = TypedArray<TypedArrayKind::Uint8Array>(runtime, size);
                  arr.updateUnsafe(runtime, (uint8_t*)mat.data, size * sizeof(uint8_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "uint16") {
                  auto arr = TypedArray<TypedArrayKind::Uint16Array>(runtime, size);
                  arr.updateUnsafe(runtime, (uint16_t*)mat.data, size * sizeof(uint16_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "uint32") {
                  auto arr = TypedArray<TypedArrayKind::Uint32Array>(runtime, size);
                  arr.updateUnsafe(runtime, (uint32_t*)mat.data, size * sizeof(uint32_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "int8") {
                  auto arr = TypedArray<TypedArrayKind::Int8Array>(runtime, size);
                  arr.updateUnsafe(runtime, (int8_t*)mat.data, size * sizeof(int8_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "int16") {
                  auto arr = TypedArray<TypedArrayKind::Int16Array>(runtime, size);
                  arr.updateUnsafe(runtime, (int16_t*)mat.data, size * sizeof(int16_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "int32") {
                  auto arr = TypedArray<TypedArrayKind::Int32Array>(runtime, size);
                  arr.updateUnsafe(runtime, (int32_t*)mat.data, size * sizeof(int32_t));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "float32") {
                  auto arr = TypedArray<TypedArrayKind::Float32Array>(runtime, size);
                  arr.updateUnsafe(runtime, (float*)mat.data, size * sizeof(float));
                  value.setProperty(runtime, "buffer", arr);
                } else if(type == "float64") {
                  auto arr = TypedArray<TypedArrayKind::Float64Array>(runtime, size);
                  arr.updateUnsafe(runtime, (double*)mat.data, size * sizeof(double));
                  value.setProperty(runtime, "buffer", arr);
                }

                return value;
      });
    } else if (propName == "createObject") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "createObject"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::create(runtime, arguments, count);
      });
    }
  else if (propName == "toJSValue") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "toJSValue"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::convertToJSI(runtime, arguments, count);
      });
    }
  else if (propName == "copyObjectFromVector") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::copyObjectFromVector(runtime, arguments, count);
      });
    }
  else if (propName == "addObjectToVector") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "addObjectToVector"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {

          FOCV_Object::addObjectToVector(runtime, arguments, count);

          return jsi::Value(true);
      });
    }
  else if (propName == "invoke") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "invoke"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Function::invoke(runtime, arguments, count);
      });
  } else if (propName == "clearBuffers") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "clearBuffers"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
          std::set<std::string> ids_to_keep;

          if (count > 0) {
            auto array = arguments[0].asObject(runtime).asArray(runtime);
            auto length = array.length(runtime);
            for (size_t i = 0; i < length; i++) {
              auto id = array.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime);
              ids_to_keep.insert(id);
            }
          }

          FOCV_Storage::clear(ids_to_keep);
          return true;
      });
  } else if (propName == "siftDetect") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "siftDetect"), 2,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
          auto origImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
          auto origImageRaw = *FOCV_Storage::get<cv::Mat>(origImageId);

          auto maskImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);
          auto maskImageRaw = *FOCV_Storage::get<cv::Mat>(maskImageId);

          cv::Mat origImageResized;// = origImageRaw;
          cv::resize(origImageRaw, origImageResized, cv::Size(800, 600));

          cv::Mat origImageBW;
          cv::cvtColor(origImageResized, origImageBW, cv::COLOR_BGRA2GRAY);

          cv::Mat origImage = origImageBW;

          cv::Mat maskImageResized;// = maskImageRaw;
          cv::resize(maskImageRaw, maskImageResized, cv::Size(800, 600));

          cv::Mat maskImageBW;
          cv::cvtColor(maskImageResized, maskImageBW, cv::COLOR_BGRA2GRAY);

          cv::Mat maskImage;
          const int lowerBound = 10;
          const int upperBound = 255;
          cv::inRange(maskImageBW, lowerBound, upperBound, maskImage);

          cv::Mat origImageMasked;
          cv::bitwise_or(origImage, origImage, origImageMasked, maskImage);

          cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();

          std::vector<cv::KeyPoint> origKeypoints;
          cv::Mat origDescriptors;

          siftPtr->detectAndCompute(origImage, origImageMasked, origKeypoints, origDescriptors);

          jsi::Object result(runtime);

          auto keypointsId = FOCV_Storage::save(origKeypoints);
          result.setProperty(runtime, "keypoints", FOCV_JsiObject::wrap(runtime, "keypoint_vector", keypointsId));

          auto descriptorsId = FOCV_Storage::save(origDescriptors);
          result.setProperty(runtime, "descriptors", FOCV_JsiObject::wrap(runtime, "mat", descriptorsId));

          return result;
    });
  } else if (propName == "siftDrawKeypoints") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "siftDrawKeypoints"), 2,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
          auto origImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
          auto origImageRaw = *FOCV_Storage::get<cv::Mat>(origImageId);

          auto origKeypointsId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);
          auto origKeypoints = *FOCV_Storage::get<std::vector<cv::KeyPoint>>(origKeypointsId);

          cv::Mat output;
          drawKeypoints(origImageRaw, origKeypoints, output, cv::Scalar::all(-1));

           cv::Mat result;
           cv::cvtColor(output, result, cv::COLOR_BGRA2RGBA);

           auto id = FOCV_Storage::save(result);
           return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  } else if (propName == "siftCompare") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "siftCompare"), 3,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Value {
          auto testImageId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
          auto testImageRaw = *FOCV_Storage::get<cv::Mat>(testImageId);

          auto origDescriptorsId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);
          auto origDescriptors = *FOCV_Storage::get<cv::Mat>(origDescriptorsId);

          auto origKeypointsId = FOCV_JsiObject::id_from_wrap(runtime, arguments[2]);
          auto origKeypoints = *FOCV_Storage::get<std::vector<cv::KeyPoint>>(origKeypointsId);

          cv::Mat testImageBW;
          cv::cvtColor(testImageRaw, testImageBW, cv::COLOR_RGBA2GRAY);

          cv::Mat testImage = testImageBW;
          //cv::resize(testImageBW, testImage, cv::Size(1024, 768));

        std::vector<cv::KeyPoint> testKeypoints;
        cv::Mat testDescriptors;
        cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
        siftPtr->detectAndCompute(testImage, cv::noArray(), testKeypoints, testDescriptors);

          drawKeypoints(testImage, testKeypoints, testImage, cv::Scalar::all(-1));

        cv::Ptr<cv::flann::KDTreeIndexParams> indexParams(
          new cv::flann::KDTreeIndexParams(5));
        cv::Ptr<cv::flann::SearchParams> searchParams(
          new cv::flann::SearchParams(50));
        cv::Ptr<cv::DescriptorMatcher> matcher(
          new cv::FlannBasedMatcher(indexParams, searchParams));
        std::vector<std::vector<cv::DMatch>> knnMatches;

        if (testDescriptors.empty()) {
          cv::Mat output;
          cv::cvtColor(testImageRaw, output, cv::COLOR_BGRA2RGBA);
          auto id = FOCV_Storage::save(output);
          return FOCV_JsiObject::wrap(runtime, "mat", id);
        }

        if(origDescriptors.type()!=CV_32F) {
            origDescriptors.convertTo(origDescriptors, CV_32F);
        }

        if(testDescriptors.type()!=CV_32F) {
            testDescriptors.convertTo(testDescriptors, CV_32F);
        }
        matcher->knnMatch(origDescriptors, testDescriptors, knnMatches, 2, cv::noArray());

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> goodMatches;
        for (size_t i = 0; i < knnMatches.size(); i++) {
          if (!knnMatches[i].empty()) {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
              goodMatches.push_back(knnMatches[i][0]);
            }
          }
        }

        if (goodMatches.size() < 50) {
          cv::Mat output;
          cv::cvtColor(testImage, output, cv::COLOR_BGRA2RGBA);
          auto id = FOCV_Storage::save(output);
          return FOCV_JsiObject::wrap(runtime, "mat", id);
        }

        std::vector<Point2f> orig;
        std::vector<Point2f> test;

        for( size_t i = 0; i < goodMatches.size(); i++ )
        {
          //-- Get the keypoints from the good matches
          orig.push_back(origKeypoints[goodMatches[i].queryIdx].pt);
          test.push_back(testKeypoints[goodMatches[i].trainIdx].pt);
        }
        Mat H = cv::findHomography(orig, test, cv::RANSAC);

        if (H.empty()) {
          cv::Mat output;
          cv::cvtColor(testImageRaw, output, cv::COLOR_BGRA2RGBA);
          auto id = FOCV_Storage::save(output);
          return FOCV_JsiObject::wrap(runtime, "mat", id);
        }

        std::vector<Point2f> origCorners(4);
        // origCorners[0] = Point2f(0, 0);
        // origCorners[1] = Point2f((float)testImage.cols, 0 );
        // origCorners[2] = Point2f((float)testImage.cols, (float)testImage.rows );
        // origCorners[3] = Point2f(0, (float)testImage.rows );
        origCorners[0] = Point2f(0, 0);
        origCorners[1] = Point2f((float)testImage.rows, 0 );
        origCorners[2] = Point2f((float)testImage.rows, (float)testImage.cols );
        origCorners[3] = Point2f(0, (float)testImage.cols );
        std::vector<Point2f> testCorners(4);

        perspectiveTransform(origCorners, testCorners, H);

        cv::Mat output;
        cv::cvtColor(testImageRaw, output, cv::COLOR_BGRA2RGBA);
          cv::line(output,
                   testCorners[0],// + Point2f((float)testImage.cols, 0),
                   testCorners[1] ,//+ Point2f((float)testImage.cols, 0),
                   Scalar( 128,128,128), 4 );
          cv::line(output,
                   testCorners[1] ,//+ Point2f((float)testImage.cols, 0),
                   testCorners[2] ,//+ Point2f((float)testImage.cols, 0),
                   Scalar( 128,128,128), 4 );
          cv::line(output,
                   testCorners[2] ,//+ Point2f((float)testImage.cols, 0),
                   testCorners[3] ,//+ Point2f((float)testImage.cols, 0),
                   Scalar( 128,128,128), 4 );
          cv::line(output,
                   testCorners[3] ,//+ Point2f((float)testImage.cols, 0),
                   testCorners[0] ,//+ Point2f((float)testImage.cols, 0),
                   Scalar( 128,128,128), 4 );
          float perc = std::round((static_cast<float>(goodMatches.size())/static_cast<float>(origKeypoints.size()))*100.0);
          cv::String text = cv::format("Matches: %zu / %zu (%f)", goodMatches.size(), origKeypoints.size(), perc);
          cv::putText(output,
                     //std::format("Matches: {}", goodMatches.size()),
                      text,
                      cv::Point(10,400),
cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            2);
          cv::String text1 = cv::format("KNN Matches: %zu", knnMatches.size());
          cv::putText(output,
                     //std::format("Matches: {}", goodMatches.size()),
                      text1,
                      cv::Point(10,500),
cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            2);

          /*
        cv::Mat output;
        drawMatches(testImage, origKeypoints, testImage, testKeypoints, goodMatches,
        output, cv::Scalar::all(-1), cv::Scalar::all(-1));
          */

          //cv::Mat result;
          //cv::cvtColor(testImageRaw, output, cv::COLOR_BGRA2RGBA);
          auto id = FOCV_Storage::save(output);
          return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  }

  return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> OpenCVPlugin::getPropertyNames(jsi::Runtime& runtime) {
    std::vector<jsi::PropNameID> result;

    result.push_back(jsi::PropNameID::forAscii(runtime, "frameBufferToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "base64ToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "matToBuffer"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "createObject"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "toJSValue"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "invoke"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "clearBuffers"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "siftDetect"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "siftDrawKeypoints"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "siftCompare"));

    return result;
}
