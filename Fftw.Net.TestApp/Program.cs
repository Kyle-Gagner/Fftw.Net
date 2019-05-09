using System;
using Fftw.Net;

namespace Fftw.Net.TestApp
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("FFTW Test:");
            var foo = FftwPlan.Dft(4);
            var bar = foo.In.Span;
            bar[0] =  1;
            bar[1] =  0;
            bar[2] =  0;
            bar[3] = -1;
            bar[4] = -1;
            bar[5] =  0;
            bar[6] =  0;
            bar[7] =  1;
            foo.Execute();
            PrintArry(foo.Out);
        }
        static void PrintArry(FftwArray arry)
        {
            var foo = arry.Span;
            for (int n = 0; n < arry.Length; n += 2)
                Console.WriteLine($"<{foo[n]}, {foo[n+1]}>");
        }
    }
}
