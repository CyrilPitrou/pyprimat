	DOUBLE PRECISION FUNCTION EX(X)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	Exponential function with underflow precaution.
C
C	Called by several subroutines
C
C	x=argument
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)

C-----For compatibility with old VAX/VMS systems
	if (x.gt.88.029d0) then        !In danger of overflow.
	  ex = dexp(88.029d0)
	else
	  if (x.lt.-88.722d0) then     !In danger of underflow.
	    ex = 0.d0
	  else                         !Value of x in allowed range.
	    ex = dexp(x)
	  endif
	endif

	RETURN
	END


