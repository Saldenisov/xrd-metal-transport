#ifndef SAXSScoreMode_h
#define SAXSScoreMode_h 1

#include <cstdlib>
#include <cstring>

// Batch inverse runs need only integer primary-photon detector counts. Keep
// the original ntuple mode available for event-history diagnostics.
inline bool SAXSImageOnly()
{
  static const bool enabled = [] {
    const char* value = std::getenv("SAXS_IMAGE_ONLY");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

inline int SAXSImagePixels()
{
  const char* value = std::getenv("SAXS_IMAGE_PIXELS");
  return value ? std::atoi(value) : 1000;
}

inline double SAXSImagePitchUm()
{
  const char* value = std::getenv("SAXS_IMAGE_PITCH_UM");
  return value ? std::atof(value) : 100.0;
}

#endif
