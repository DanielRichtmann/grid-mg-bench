/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/multigrid/Profiler.h

    Copyright (C) 2015-2018

    Author: Daniel Richtmann <daniel.richtmann@ur.de>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
/*  END LEGAL */
#pragma once

#include <set>

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(Rework);

///////////////////////////////////////////////////////////////////////////////
//                 These structs define how the output looks                 //
///////////////////////////////////////////////////////////////////////////////

struct Sec {
  Sec(double usec) { seconds = usec / 1.0e6; }
  double seconds;
};

inline std::ostream& operator<<(std::ostream& stream, const Sec&& s) {
  // if(s.seconds != 0.0) stream << std::scientific << "t[s] = " << s.seconds;
  // if(s.seconds != 0.0) stream << std::scientific << "t = " << s.seconds << " s";
  if(s.seconds != 0.0) stream << std::scientific << s.seconds << " s";
  return stream;
}

struct Perf {
  Perf(double flop, double usec) {
    flopPerSecond = (usec == 0.0) ? 0.0 : flop / usec * 1.0e6;
  }
  double flopPerSecond;
};

inline std::ostream& operator<<(std::ostream& stream, const Perf&& p) {
  // if(p.flopPerSecond != 0.0) stream << std::scientific << "P[F/s] = " << p.flopPerSecond;
  // if(p.flopPerSecond != 0.0) stream << std::scientific << "P = " << p.flopPerSecond << " F/s";
  if(p.flopPerSecond != 0.0) stream << std::scientific << p.flopPerSecond << " F/s";
  return stream;
}

struct Traffic {
  Traffic(double byte, double usec) {
    bytePerSecond = (usec == 0.0) ? 0.0 : byte / usec * 1.0e6;
  }
  double bytePerSecond;
};

inline std::ostream& operator<<(std::ostream& stream, const Traffic&& t) {
  // if(t.bytePerSecond != 0.0) stream << std::scientific << "T[B/s] = " << t.bytePerSecond;
  // if(t.bytePerSecond != 0.0) stream << std::scientific << "T = " << t.bytePerSecond << "B/s";
  if(t.bytePerSecond != 0.0) stream << std::scientific << t.bytePerSecond << " B/s";
  return stream;
}

struct Intensity {
  Intensity(double intensity) { flopPerByte = intensity; }
  double flopPerByte;
};

inline std::ostream& operator<<(std::ostream& stream, const Intensity&& i) {
  // if(i.flopPerByte != 0.0) stream << std::scientific << "I[F/B] = " << i.flopPerByte;
  // if(i.flopPerByte != 0.0) stream << std::scientific << "I = " << i.flopPerByte << " F/B";
  if(i.flopPerByte != 0.0) stream << std::scientific << i.flopPerByte << " F/B";
  return stream;
}

struct TimeFraction {
  TimeFraction(double usec, double usecTotal) {
    percent = (usecTotal == 0.0)? 0.0 : 100 * usec / usecTotal;
  }
  double percent;
};

inline std::ostream& operator<<(std::ostream& stream, const TimeFraction&& tf) {
  // if(tf.percent != 0.0) stream << std::fixed << "F[%] = " << tf.percent;
  // if(tf.percent != 0.0) stream << std::fixed << "F = " << tf.percent << " %";
  if(tf.percent != 0.0) stream << std::fixed << tf.percent << " %";
  return stream;
}

struct Calls {
  Calls(double calls) { number = calls; }
  double number;
};

inline std::ostream& operator<<(std::ostream& stream, const Calls&& c) {
  // stream << std::scientific << "C[#] = " << c.number;
  // stream << std::scientific << "C = " << c.number << " #";
  stream << std::scientific << c.number << " x";
  return stream;
}

/////////////////////////////////////////////////////////////////////////////
//  Now we have to devise a way to collect the performance data elegantly  //
/////////////////////////////////////////////////////////////////////////////

class MGPerfData {
public:
  // these take full values of flop,byte,calls between start and stop
  void   Perf(double flop)    { flop_ += flop; }
  void   Traffic(double byte) { byte_ += byte; }
  void   Calls(double calls)  { calls_ += calls; }
  double Perf()      const    { return flop_; }
  double Traffic()   const    { return byte_; }
  double Intensity() const    { return (byte_ != 0.0) ? flop_ / byte_ : 0.; }
  double Calls()     const    { return calls_; }

private:
  double flop_;
  double byte_;
  double calls_;
};

struct MGProfileResult {
  MGPerfData pd;
  GridTime   t;
};

std::map<std::string, MGProfileResult> addPrefix(std::map<std::string, MGProfileResult> const& in,
                                                 std::string const&                            prefix) {
  std::map<std::string, MGProfileResult> out;
  for(auto const& elem : in) {
    auto key = prefix + elem.first;
    out[key] = elem.second;
  }
  return out;
}

class Profiler {
private:
  struct MGProfileData {
    MGPerfData    pd;
    GridStopWatch sw;
  };

  std::string                          prefix_;
  std::map<std::string, MGProfileData> val_;

public:
  void Start(std::string const& name, double flop = 0.0, double byte = 0.0) {
    if(!name.empty()) {
      val_[name].pd.Perf(flop);
      val_[name].pd.Traffic(byte);
      val_[name].sw.Start();
      std::cout << GridLogDebug << name << ".Start" << std::endl;
    }
  }

  void Stop(std::string const& name, double calls = 1) {
    if(val_[name].sw.isRunning()) {
      val_[name].sw.Stop();
      val_[name].pd.Calls(calls);
      std::cout << GridLogDebug << name << ".Stop" << std::endl;
    }
  }

  void ResetAll() { val_.clear(); }

  void AddPrefix(std::string const& prefix) {
    if(prefix_.empty()) {
      prefix_ = prefix;
    } else {
      prefix_ += "." + prefix;
    }
  }

  void ResetPrefix() {
    prefix_ = "";
  }

  std::map<std::string, MGProfileResult> GetResults(std::string const& prefix = "") const {
    std::map<std::string, MGProfileResult> results;
    for(auto const& elem : val_) {
      auto index        = prefix + elem.first;
      results[index].pd = elem.second.pd;
      results[index].t  = elem.second.sw.Elapsed();
    }
    return results;
  }

  std::map<std::string, GridTime> GetTimings() const {
    std::map<std::string, GridTime> timings;
    for(auto const& elem : GetResults()) {
      timings[elem.first] = elem.second.t;
    }
    return timings;
  }

  std::map<std::string, MGPerfData> GetPerformanceData() const {
    std::map<std::string, MGPerfData> perfData;
    for(auto const& elem : GetResults()) {
      perfData[elem.first] = elem.second.pd;
    }
    return perfData;
  }
};

void prettyPrintProfiling(std::string const& prefix,
                          std::map<std::string, MGProfileResult> const& results,
                          GridTime totalTime = GridTime(0),
                          bool verbose = false) {
  double       usecTotal = static_cast<double>(totalTime.count());
  auto         cf        = std::cout.flags();
  auto         p         = std::cout.precision();
  unsigned int width     = 0;

  for(auto const& elem : results)
    width = std::max(width, static_cast<unsigned int>(elem.first.length()));

  typedef std::function<bool(
    std::pair<std::string, MGProfileResult>,
    std::pair<std::string, MGProfileResult>)>
    Compare;

  std::set<std::pair<std::string, MGProfileResult>, Compare> sortedResults(
    results.begin(),
    results.end(),
    [](
      std::pair<std::string, MGProfileResult> elem1,
      std::pair<std::string, MGProfileResult> elem2) {
      return elem1.second.t.count() < elem2.second.t.count();
    });

  for(auto const& elem : sortedResults) {
    double     usec = static_cast<double>(elem.second.t.count());
    MGPerfData mgpd = elem.second.pd;

    if(!usec)
      continue;

    // clang-format off
    if(verbose) {
      std::cout << GridLogPerformance
                << prefix                                       << (prefix.empty() ? "" : " : ")
                << std::setw(width)
                << elem.first                                   << " : "
                << Sec(usec)                                    << " "
                << Calls(mgpd.Calls())                          << " "
                << Intensity(mgpd.Intensity())                  << " "
                << Perf(mgpd.Perf() * mgpd.Calls(), usec)       << " "
                << Traffic(mgpd.Traffic() * mgpd.Calls(), usec) << " "
                << TimeFraction(usec, usecTotal)
                << std::endl;
    } else {
      std::cout << GridLogPerformance
                << prefix     << (prefix.empty() ? "" : " : ")
                << std::setw(width)
                << elem.first << " : "
                << Sec(usec) << " "
                << Calls(mgpd.Calls())
                << ((usecTotal != 0.0) ? " " : "")
                << TimeFraction(usec, usecTotal)
                << std::endl;
    }
    // clang-format on
  }

  std::cout.flags(cf);
  std::cout.precision(p);
}

class Profileable {
protected:
  Profiler prof_;

public:
  void ResetProfile() { prof_.ResetAll(); }

  std::map<std::string, MGProfileResult> GetProfile(std::string const& prefix = "") const {
    return prof_.GetResults(prefix);
  }
};

NAMESPACE_END(Rework);
NAMESPACE_END(Grid);
