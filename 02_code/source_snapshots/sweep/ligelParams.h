#ifndef LIGELPARAMS_H
#define LIGELPARAMS_H

// simulation parameters

struct SimParams {
	int LX;
	int LY;
	int ig;
	double f;
	double q;
	double ep;
	double epini;
	double drep;
	double P1;
	double P2;
	double dt;
	double dtx;
	double dx;
	double dy;
	double CH0;
	double CH1;
	double CHS;
	double C0;
	double LAMDAV;
	double AZ0;
	double FA0;
	double uss;
	double vss;
	double wss;
	double phi;
	int2 rn_offset[9];
	int2 um_offset_noflux[9];
	int2 um_offset_periodic[9];
	int TargetWave_y;
	double distance;
	double Repulsion_wall;
	int maxFilamentlen;

	// [FchemoCoupling] Substrate-memory osmotic coupling (Y-B 2007 Eq. 22 extension).
	// Adds a term  -CHM * M * phi * (1-phi)  to the Flory-Huggins free energy,
	// parallel in form to Y-B's catalyst-hydrating term  -CHS * v * phi * (1-phi).
	// Yields a +CHM * w * M contribution to osmotic pressure pm (Y-B Eq. 25 extended),
	// hence an emergent body force ~CHM * w * grad(M) at gel interior nodes via
	// Y-B's existing nodal-force assembly (Eq. 48) in calNodesVelocityD.
	//
	// Y-B Table II: CHS = 0.25 (=chi*).  Sasaki 2003 (Y-B ref 21) bounds CHM << CHS.
	// Recommended scan: CHM in [1e-4, 1e-2], i.e. CHM/CHS in [0.001, 0.1].
	// CHM = 0.0 (default) recovers original Y-B model bit-for-bit.
	double CHM;
};


#endif
