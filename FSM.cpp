#include <iostream>
#include <string>

using namespace std;

enum State { S0, S1, S2, S3 };

bool accepts(const string &input) {
    State current = S0; // start state
    
    for(char c : input) {
        switch(current) {
            case S0:
                if(c == 'a') current = S0;
                else if(c == 'b') current = S1;
                else return false; // invalid input
                break;
            case S1:
                if(c == 'a') current = S2;
                else if(c == 'b') current = S1;
                else return false;
                break;
            case S2:
                if(c == 'a') current = S0;
                else if(c == 'b') current = S3;
                else return false;
                break;
            case S3:
                if(c == 'a') current = S2;
                else if(c == 'b') current = S1;
                else return false;
                break;
        }
    }
    
    return current == S3; // accept if in final state
}

int main() {
    string testStr;
    cout << "Enter a string over {a,b}: ";
    cin >> testStr;
    
    if(accepts(testStr))
        cout << "String accepted: ends with 'bab'" << endl;
    else
        cout << "String rejected: does not end with 'bab'" << endl;

    return 0;
}