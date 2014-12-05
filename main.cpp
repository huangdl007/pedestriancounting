#include <iostream>
#include "HOG.h"

using namespace std;

int main()
{
	HOG hog;

	cout << hog.getPeopleDetector().size();
	return 0;
}