#include <set>
#include <iostream>

using namespace std;

int main(){
    
    set<int> filtered;
    filtered.insert(1);
    filtered.insert(2);
    filtered.insert(3);
    filtered.insert(4);
    filtered.insert(5);
    set<int>::iterator it;
    for (it = filtered.begin(); it != filtered.end(); ++it){
        int l = *it;
        cout << l << endl;
    }
    
}