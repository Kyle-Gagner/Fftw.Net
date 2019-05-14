# Fftw.Net
Cross platform .NET wrappers for FFTW

## Status
Fftw.Net is pre-alpha and actively developed.

## Project Goals
* Create a P/Invoke wrapper targetting .NET Standard, .NET Core, and .NET Framework
* Support all FFTW functions minus the 64-bit interface
* Supply a safe API to the greatest extent possible (no resource leaks, crashes, etc.)
* Follow good .NET patterns in the object model and in resource use, etc.
* Reach a 1.0 release with a promise of ongoing backward compatibility between subsequent releases
* Create a Nuget package and publish on NuGet.org
* Investigate distributing native runtimes, subject to licensing and other considerations
* Support minimally:
    * Windows x64
    * Linux x64
* Potential platform goals include
    * macOS
    * Windows & Linux x86
    * Linux on ARM architectures