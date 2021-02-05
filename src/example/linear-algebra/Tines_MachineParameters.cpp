#include "Tines.hpp"
#include "Tines_Interface.hpp"

int main(int argc, char **argv) {

#if defined(TINES_ENABLE_TPL_LAPACKE_ON_HOST)
  printf("LAPACKE Machine Parameters\n");
  printf("==========================\n");

  { /// single precision
    const float eps = LAPACKE_slamch('E');
    const float sfmin = LAPACKE_slamch('S');
    const int base = LAPACKE_slamch('B');
    const float prec = LAPACKE_slamch('P');

    printf("single precision\n");
    printf("  eps   %e\n", eps);
    printf("  sfmin %e\n", sfmin);
    printf("  base  %d\n", base);
    printf("  prec  %e\n", prec);
    printf("\n\n");
  }
  { /// double precision
    const double eps = LAPACKE_dlamch('E');
    const double sfmin = LAPACKE_dlamch('S');
    const int base = LAPACKE_dlamch('B');
    const double prec = LAPACKE_dlamch('P');

    printf("double precision\n");
    printf("  eps   %e\n", eps);
    printf("  sfmin %e\n", sfmin);
    printf("  base  %d\n", base);
    printf("  prec  %e\n", prec);
    printf("\n\n");
  }
#endif

  printf("Tines Machine Parameters\n");
  printf("========================\n");

  { /// single precision
    using ats = Tines::ats<float>;
    printf("single precision\n");
    printf("  eps   %e\n", ats::epsilon());
    printf("  sfmin %e\n", ats::sfmin());
    printf("  base  %d\n", ats::base());
    printf("  prec  %e\n", ats::prec());
    printf("\n\n");
  }
  { /// single precision
    using ats = Tines::ats<double>;
    printf("double precision\n");
    printf("  eps   %e\n", ats::epsilon());
    printf("  sfmin %e\n", ats::sfmin());
    printf("  base  %d\n", ats::base());
    printf("  prec  %e\n", ats::prec());
    printf("\n\n");
  }
  return 0;
}
