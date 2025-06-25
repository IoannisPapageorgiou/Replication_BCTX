#include <iostream>
#include <string>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <algorithm>  // needed for sort
#include <iomanip>    // needed for setw
#include <vector>
#include <random>
#include <map>
#include <time.h> 
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

const int m = 2; // alphabet size
const int D = 10; // max depth
long double beta = 0.5; // prior parameter
const long double alpha = pow((1.0 - beta), (1.0 / (m - 1.0)));
const short k_max = 1;

const short ar_p = 2; //order of AR models
//const double e_n = 0.1346; // variance of Gaussian noise, i.e. en=sigma^2


Matrix<double, ar_p + 1, 1> mu_0;       //mean vector of Gaussian prior for parameters
Matrix<double, ar_p + 1, ar_p + 1> sigma_0; // cov matrix of Gaussian prior for parameters, det these to values at start of main

long double tau = 1.0;
long double lambda = 1.0;



typedef vector <vector <short > > matrix;
const vector <short> zeros(m, 0);
const vector <short> ones(m, 1);

const vector<double> init_s2(ar_p + 1, 0);//used to initialize sums s2 and s3 to all zero vector and matrix
const vector < vector <double> > init_s3(ar_p + 1, init_s2);

class node // introduce a class for nodes
{
public:

	vector <short> s = {};        // context, maybe not needed as position in tree defines suffix, useful for debugging (insert 1 time, in show_tree 4 times, in init_tree 1 time), could use char

	int a[m] = { 0 };       // occurrences for each j in alphabet NOT NEEDED FOR AR
	//int M = 0;             // sum of aj's, can avoid storing and just use sums
	// long double p=1;    // Pe(as), work with logarithm for better accuracy
	// long double pw = 1; // can use these for debugging tests
	double le = 0;    // Here, logarithm to base 2 is used, Le(as) // can use float or double (compromise memory vs accuraccy)
	double lw = 0;    // logarithm of weighted probability/maximal probability
	vector <double> lm = { 0 };
	matrix c;



	bool leaf = 0;   // indicates if node is a leaf
	//float theta[m];  // parameters of distirbution for leaf nodes


	//node * self;    // pointer for this node, not needed because can use &node_name
	//node * par;     // pointer for parent node, not needed
	node* child[m] = { 0 };  // pointers for children nodes

	// sums for continuous process: Bs, s1,s2,s3
	int Bs = 0;
	double s1 = 0;
	vector <double> s2 = init_s2;
	vector < vector <double> > s3 = init_s3;

};



typedef vector <vector <node*> >  tree; // introduce a structure for a tree: group of node pointers
// tree[m]: store pointers of nodes at depth m

class tree2 // introduce a class for trees, grouping together the (pointer) tree and its chracteristics
{
public:

	tree t; // node pointers as above
	int d; // max depth of tree
	vector <node*> leaves; // leaves of tree !!!!AT DEPTH <= D-1!!!!
	vector <node*> pl; // nodes with only m decendants (which are necessarily leaves)
	int f = 0; // mcmc frequency

	map < vector <short>, vector <float>> theta;

};

// function prototypes
vector<short> read(string s);           // reads file and stores xn into vector
vector<short> read2(string s);           // reads file and stores xn into vector, for m > 9 and one symbol in each line
void write(vector<long double> h, string s); //writes elements of h in file with filename s

void update(tree& T, double s, vector <short> ct, vector<double> x_tilde);   // updates tree for each new symbol
void occur(node* N, double s, vector<double> x_tilde);                    // node occurrence updates
void insert(tree& T, vector <short> ct2, short ch); // inserts node with context ct2 to T, links with existing child
node* search(tree T, vector <short> ct);         // searches tree for context, returns node pointer

int show_tree(tree T);                        // prints nodes and children and return max depth of tree
int show_tree2(tree T);                         // use when nodes not deleted
int show_leaves(tree T);                       // prints leaves and returns number of leaves
int show_leaves2(tree T);
int no_leaves_vlmc(tree T);

void init_tree(tree& T);                       // some operations to initialise tree
void build_tree(tree& T, vector <double> xn);   // builds improper tree of maxdepth from sequence xn

long double rma(tree& T);                       // rma algorithm, returns mean marginal likelihood (this one takes improper tree input and also makes it proper)
long double rma2(tree& T);                      // rma algorithm, returns mean marginal likelihood (this one takes proper tree input)
long double rma3(tree& T);                      // identical to rma but does not make tree proper (less memory needed)

long double mle(tree T);                         // finds log-maximum likelihood (input is proper tree of any depth)
long double mle2(tree T);                        // finds log-maximum likelihood (input is improper tree of max depth)
long double mlevlmc(tree T);                     // finds log-max likelihoo for a general (maybe improper tree T)

void counts(vector <double> xn, tree& T);         // finds counts for proper tree T in sequence xn
void counts2(vector <double> xn, tree& T);        // finds count in maybe improper tree T

long double bic(tree& T, vector <double> xn);       // finds bic of (proper) tree T, sequence xn
long double bicvlmc(tree& T, vector <double> xn);   // find bic of maybe improper tree T, sequence xn

long double logloss(vector <double> xn, int T);   // finds log-loss when predicting from x(n-T) to xn

long double mapt(tree& T);                      // mapt algorithm, returns max prob at root: it takes improper tree input and gives the proper MAP tree (used after rma3)
long double mapt2(tree& T);                     // older version, dummy creation of nodes that might be deleted later
long double mapt3(tree& T);                     // this takes proper tree input (useful following rma algorithm which makes tree proper)
long double mapt4(tree& T);                     // does not delete elements to reallocate memory, but is faster

void kmapt_forw(tree& T, vector <node*> init);     // forward pass of kmapt, takes improper and gives improper tree, calculates vectors lm, c for each node
void kmapt_back(vector <node*> init, tree T, vector <tree>& trees); // backward pass of kmapt
void kmapt(tree& T, vector <tree>& trees, vector <node*> init, vector<double>& odds);

void rma_mapt(tree& T);                  // run rma3 and mapt, gives output mapt tree, prints leaves of tree, prior, posterior, max depth
void rma_mapt2(tree& T);                 // runs rma3 and mapt4 (faster but doesn't release memory)

void label(tree& T);
void label2(tree& T);

void deltree(tree& T);                   //deletes nodes that are marked to be deleted
tree copytree(tree T);                   // copies only non-deleted nodes to store proper pruned tree
tree copy(tree Tin);                     // makes an identical copy of a tree, so that can process 2nd while keeping the 1st unchanged

int difference(vector <short> c);                          // finds L1 distance between integer vectors
void comb_int(int n, int r, vector <int> list);            //combination with repetition from n choose r from list
void comb_initial(int n, int r, int d, vector <node*> init);   //combinations for preprocessing part (12=21, probably wrong)
void comb_initial2(int d, vector <node*> init);                  //identical with comb but for preprocessing, probably right version
void comb_initial3(int d, vector <node*> init);
void preproc(vector <node*> init);                             //does the preprocessing calcultions

void comb(int d, int k, tree& T, vector <node*> init);                // combinations for k-mapt forward pass


vector<vector<double> > cart_product(const vector<vector<double>>& v);  //combinations for doubles
vector<vector<short> > cart_product_int(const vector<vector<short>>& v);//combinations for shorts

tree2 construct(tree T);          // takes tree T and find info to construct tree2 
void makeproper(tree& T);         // takes improper tree input and makes it proper
void dictionary(map < vector <short>, node*>& dict, tree Tmax);   // creates a dictionary from Tmax


vector < vector <short>> treeid(tree2 T);                                  // finds identity of tree T
vector < vector <short>> treeid2(tree T);
void neighbours(tree2 T1, tree2 T2, vector <short>& context, int& point);
int find_neighbour(tree2 Tsmall, tree2 Tbig, vector <short>& context);


void mcmc(map< vector < vector <short>>, tree2*>& mcmc_trees, tree2 T0, map < vector <short>, node*> dict, int N);
void mcmc_jump(double p, map< vector < vector <short>>, tree2* >& mcmc_trees, vector <tree2> Tstar, short T0, map < vector <short>, node*> dict, int N, vector <double> odds);
void propose_find_ratio(map< vector < vector <short>>, tree2* >& mcmc_trees, tree2& Tc, map < vector <short>, node*> dict);
void propose(double p, map< vector < vector <short>>, tree2* >& mcmc_trees, tree2& Tc, vector <tree2> Tstar, map < vector <short>, node*> dict);
void jump(double p, map< vector < vector <short>>, tree2* >& mcmc_trees, tree2& Tc, vector <tree2> Tstar, map < vector <short>, node*> dict, vector <double> odds);
void gibbs_step(tree2& T, map < vector <short>, node*> dict); //samples thetas at leaves of tree
map < vector <short>, vector <float>> gibbs_step2(tree T, map < vector <short>, node*> dict);


tree sample(map < vector <short>, node*> dict);
void sample_trees(int N, map < vector <short>, node*> dict, map< vector < vector <short>>, tree2* >& mcmc_trees);
vector <long double> entr(int N, map < vector <short>, node*> dict);
vector <long double> entr2(int N, map < vector <short>, node*> dict);

void marg_entr(int N, map < vector <short>, node*> dict);


vector <short> generate_from_tree(tree2 T, int N, long double& entr);  //generate N samples from tree T 
vector <short> generate_from_tree2(tree T, map < vector <short>, vector <float>> theta, int N, long double& entr);
long double power_method(tree2 T, int n, map < vector <short>, node*> dict);
long double power_method2(tree T, map < vector <short>, vector <float>> theta, int n, map < vector <short>, node*> dict);

vector< short> dec2d(int r, int  d); // decimal to m-ary
vector< vector<float>> trans_matrix(tree2 T); // finds transition matrix for power method for tree T
vector< vector<float>> trans_matrix2(tree T, map < vector <short>, vector <float>> theta);
vector<float> vec_times_mat(vector <float> x, vector <vector <float>> P);

void p_est(node* N);
void post_param(tree T);


long double predict_one_step(tree T, vector <short> ct, vector<double> x_tilde);
void pred_mse(vector <double> xn, int train_size);
long double predict_multi_step(tree T, vector <short> ct, vector<double> x_tilde, short step, int N_samples);


void log_loss(vector <double> xn, int train_size); //log-loss achieved if using full-predictive distribution

default_random_engine generator(9982244441);

int main() {

	sigma_0 = sigma_0.Identity();
	sigma_0 = 1.0 * sigma_0;


	ifstream if_file("unemp.txt", ios::in);

	vector <double> xn;
	double temp_x = 0.0;
	while (if_file >> temp_x) {
		xn.push_back(temp_x);
	}


	cout << "File was read " << endl;

	pred_mse(xn, 144);

	return 0;


}








void update(tree& T, double s, vector <short> ct, vector<double> x_tilde) {

	node* temp = T[0][0]; // start from root

	occur(temp, s, x_tilde);


	for (int j = 0; j < D; j++) {

		if (temp->child[ct[j]] != NULL) { // if child node exists in tree

			temp = temp->child[ct[j]];    // move to child
			occur(temp, s, x_tilde);               // child occurred
		}

		else {                            // create children up to depth D

			vector <short> ct2 = ct;        // context of node to be created
			short ch = 0;                   // shows which child exists

			for (int k = 0; k < D - j; k++) {

				// depth = ct2.size();  // or depth = D - k ; // depth of node to be created

				//insert node with context ct2 to tree
				insert(T, ct2, ch);

				occur(T[ct2.size()].back(), s, x_tilde); // inserted nodes occurs

				ch = ct2.back();
				ct2.pop_back();

			}

			j = D + 5;  // break out of previous loop if a child doesn't exist
			temp->child[ch] = T[ct2.size() + 1].back();


		}

	}

}



void occur(node* N, double s, vector<double> x_tilde) {

	//N->a[s]++; //not further needed for continuous
	//int M = 0;
	//for (int i = 0; i < m; i++) { M = M + N->a[i]; }

	N->Bs++;

	N->s1 = N->s1 + pow(s, 2);

	for (int i = 0; i < ar_p + 1; i++) {
		N->s2[i] = N->s2[i] + s * x_tilde[i];
	}

	for (int i = 0; i < ar_p + 1; i++) {
		for (int j = 0; j < ar_p + 1; j++) {
			N->s3[i][j] = N->s3[i][j] + x_tilde[i] * x_tilde[j];
		}
	}

	//estimated probabilities will be calculated at the end for the continuous case

	//N->p = N->p * (N->a[s] - 0.5) / (0.5*m + M - 1);// work with logarithms
	//N->le = 1.0* N->le + log2(1.0* N->a[s] - 0.5) - log2(0.5*m + 1.0* M - 1.0);
	//cout << "occur" << endl;
	//cout <<pow(2, N->le) << endl;

}

void insert(tree& T, vector <short> ct2, short ch) {

	int d = ct2.size();            // node depth
	node* init = new node;        // initialise a node pointer
	T[d].push_back(init);         // add a node to tree at corresponding depth

	//T[d].back()->s = ct2;         // set context of node, if used (mainly for debugging)

	//extra initialization for s2,s3 to zero

	for (int i = 0; i < ar_p + 1; i++) { init->s2.push_back(0); }
	for (int i = 0; i < ar_p + 1; i++) {
		init->s3.push_back(init->s2);
	}




	if (d == D) {

		T[d].back()->leaf = 1;   // no children if leaf
		//cout << "insert leaf, ";
	}

	else {                     // set address of children given by ch

		T[d].back()->child[ch] = T[d + 1].back();
		//cout << "insert node with child, ";

	}


}



node* search(tree T, vector <short> ct) {

	node* temp = T[0][0]; // start from root

	for (int j = 0; j < D; j++) {

		if (temp->child[ct[j]] != NULL) { // if child node exists in tree
			temp = temp->child[ct[j]];    // move to child
		}

		else {                            // create children up to depth D
			return 0;
		}

	}

	return temp;
}

void build_tree(tree& T, vector <double> xn) { // build improper tree T from sequence xn

	// update for each sequence symbol
	for (int i = D; i < xn.size(); i++) {
		//cout << xn[i] << endl; //prints sequence

		double s = xn[i];          // current symbol

		vector<double> x_tilde;     //continuous context needed for sums
		x_tilde.push_back(1.0);
		for (int j = 1; j < ar_p + 1; j++) {
			x_tilde.push_back(xn[i - j]);
		}

		vector <short> ct(D);     // current context

		//cout << endl << "symbol " << i << ", with context ";

		for (int j = 0; j < D; j++) { //here use QUANTISED version for disrete context

			//ct[j] = xn[i - j - 1];  // sets context
			//cout << ct[j];   // prints context
			if (xn[i - j] > xn[i - j - 1]) {
				ct[j] = 1;
			}
			else { ct[j] = 0; }
		}
		//cout << endl;

		update(T, s, ct, x_tilde);

	}

	cout << "Tree was built" << endl;
}

int show_tree(tree T) { // uses node class with contexts, returns max depth of tree

	int maxdepth = 0;
	int n_leaves = 0;

	for (int u = 0; u < D + 1; u++) {
		for (int v = 0; v < T[u].size(); v++)
		{

			if (T[u][v]->leaf == 1) { n_leaves++; }
			cout << u << v << " node ct is ";
			for (int m = 0; m < T[u][v]->s.size(); m++)
			{
				cout << T[u][v]->s[m];
			}

			//cout << " and has occurred " << T[u][v]->M << " times" << endl;
			cout << " has children:  ";
			for (short d = 0; d < m; d++)
			{
				if (T[u][v]->child[d] != NULL) {
					node* temp123 = T[u][v]->child[d];
					cout << d << " with ct ";
					for (int k = 0; k < (temp123->s.size()); k++) {
						cout << temp123->s[k];
					}
					cout << " and ";

				}
			}

			cout << " no more" << endl;
			maxdepth = u;
		}
	}
	cout << "max depth is  " << maxdepth << endl;
	cout << "number of leaves is " << n_leaves << endl;
	return n_leaves;
}

int show_tree2(tree T) { // uses node class with contexts, returns max depth of tree, use when nodes NOT deleted from mapt tree

	int maxdepth = 0;
	int n_leaves = 0;

	for (int u = 0; u < D + 1; u++) {

		int check = 0;

		for (int v = 0; v < T[u].size(); v++) {

			if (T[u][v]->a[0] > -1)
			{

				if (T[u][v]->leaf == 1) { n_leaves++; }
				cout << u << v << " node ct is ";
				for (int m = 0; m < T[u][v]->s.size(); m++)
				{
					cout << T[u][v]->s[m];
				}

				//cout << " and has occurred " << T[u][v]->M << " times" << endl;
				cout << " has children:  ";
				for (short d = 0; d < m; d++)
				{
					if (T[u][v]->child[d] != NULL) {
						node* temp123 = T[u][v]->child[d];
						cout << d << " with ct ";
						for (int k = 0; k < (temp123->s.size()); k++) {
							cout << temp123->s[k];
						}
						cout << " and ";

					}
				}

				cout << " no more" << endl;
				maxdepth = u;
			}

			else { check++; }

		}

		if (check == T[u].size()) {
			cout << "max depth is  " << maxdepth << endl;
			cout << "number of leaves is " << n_leaves << endl;
			return n_leaves;
		}

	}
	cout << "max depth is  " << maxdepth << endl;
	cout << "number of leaves is " << n_leaves << endl;
	return n_leaves;
}

int show_leaves(tree T) { // uses node class with contexts, returns number of leaves

	int n_leaves = 0;
	int maxdepth = 0;

	for (int u = 0; u < D + 1; u++) {
		for (int v = 0; v < T[u].size(); v++)
		{

			if (T[u][v]->leaf == 1) {

				cout << u << v << " node ct is ";
				for (int m = 0; m < T[u][v]->s.size(); m++)
				{
					cout << T[u][v]->s[m];
				}

				//cout << " and has occurred " << T[u][v]->M << " times" << endl;
				cout << endl;
				n_leaves++;

			}

			maxdepth = u;
		}
	}

	cout << "max depth is  " << maxdepth << endl;
	cout << "number of leaves is " << n_leaves << endl;
	return n_leaves;

}

int show_leaves2(tree T) { // uses node class with contexts, returns number of leaves

	int n_leaves = 0;
	int maxdepth = 0;

	for (int u = 0; u < D + 1; u++) {

		int check = 0;

		for (int v = 0; v < T[u].size(); v++)
		{
			if (T[u][v]->a[0] > -1) {

				if (T[u][v]->leaf == 1) {

					cout << u << v << " node ct is ";
					for (int m = 0; m < T[u][v]->s.size(); m++)
					{
						cout << T[u][v]->s[m];
					}

					//cout << " and has occurred " << T[u][v]->M << " times" << endl;
					cout << endl;
					n_leaves++;

				}

				maxdepth = u;
			}

			else { //node doesn't exist in reality
				check++;
			}
		}

		if (check == T[u].size()) {
			cout << "max depth is  " << maxdepth << endl;
			cout << "number of leaves is " << n_leaves << endl;
			return n_leaves;
		}

	}

	cout << "max depth is  " << maxdepth << endl;
	cout << "number of leaves is " << n_leaves << endl;
	return n_leaves;

}

int no_leaves_vlmc(tree T) { // uses node class with contexts, returns number of nodes with parameter vectors

	int n_leaves = 0;
	int maxdepth = 0;

	for (int u = 0; u < D + 1; u++) {
		for (int v = 0; v < T[u].size(); v++)
		{

			if (T[u][v]->leaf == 1) {

				n_leaves++;

			}

			else {

				int check = 0;

				for (int ch = 0; ch < m; ch++) {
					if (T[u][v]->child[ch] != NULL) {
						check++;
					}
				}

				if (check < m) {

					n_leaves++;
				}


			}

			maxdepth = u;
		}
	}

	cout << "max depth is  " << maxdepth << endl;
	cout << "number of leaves is " << n_leaves << endl;
	return n_leaves;

}




void init_tree(tree& T) {

	node* root = new node;  // initialise a root node pointer


	// initially keep only root
	for (int d = 0; d < D + 1; d++) {
		T[d].pop_back();
		//T[d].reserve(pow(m, d));    // not needed in this implementation as tree stores vector of node pointers
		//  cout << " depth "<< d <<" capacity is " << T[d].capacity() << endl;
	}

	T[0].push_back(root);
	//T[0][0]->s = { 9 };            // arbitrary context for root, if contexts used

	if (D == 0) {
		T[0][0]->leaf = 1;        // if only root: iid case
	}

}

long double rma(tree& T) {                   // algorithm takes improper tree finds mean marg lik and makes tree proper

	for (int d = D; d > -1; d--) {           // loop over levels

		//int length = T[d].size();           // nodes of each level might change when making the tree proper, but this affects previous level

		for (int k = 0; k < T[d].size(); k++) {  // loop over initially existing nodes of each level

			if (d == D) {                   // if at max depth, node is a leaf
				T[d][k]->lw = T[d][k]->le;
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (short ch = 0; ch < m; ch++) {

					if (T[d][k]->child[ch] != NULL) {           // if child #ch exists
						sum = sum + T[d][k]->child[ch]->lw;     // calculate sum of Le(s)
						//prod = prod * T[d][k]->child[ch]->pw;
					}

					else {                                      // if child #ch does not exist, create it to make tree proper

						node* init = new node;
						T[d][k]->child[ch] = init;             // connect child with parent node
						T[d + 1].push_back(init);              // store at appropriate tree depth
						init->leaf = 1;                        // denote it leaf
						//init->s = T[d][k]->s;                  // if contexts used
						//init->s.push_back(ch);                 // if contexts used


					}

				}

				//calculate weighted log-prob in two cases as explained in notes for numerical precision

				double delta = T[d][k]->le - sum + log2(beta) - log2(1.0 - beta);
				if (delta < 30) {

					T[d][k]->lw = log2(1.0 - beta) + sum + log2(1.0 + pow(2.0, delta));
					//cout << pow(2, T[d][k]->lw) << endl;
				}
				else {
					T[d][k]->lw = log2(beta) + T[d][k]->le + log2(exp(1)) * (pow(2.0, -delta) - pow(2.0, -2.0 * delta - 1));
					//cout << pow(2, T[d][k]->lw) << endl;
				}
				//T[d][k]->pw = beta * T[d][k]->p + (1.0 - 1.0*beta)*prod;
			}
		}
	}

	cout << "Mean marginal likelihood is " << pow(2.0, T[0][0]->lw) << " and has log " << T[0][0]->lw << endl;
	return T[0][0]->lw;              //output value of weighted prob Pw at root
}

long double rma2(tree& T) {                        // algorithm takes proper tree and finds mean marg lik

	for (int d = D; d > -1; d--) {                // loop over levels


		for (int k = 0; k < T[d].size(); k++) {  // loop over nodes of each level

			if (T[d][k]->leaf == 1) {                       // if node is a leaf
				T[d][k]->lw = T[d][k]->le;
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (short ch = 0; ch < m; ch++) {  // for proper tree all children exist if node is not a leaf

					sum = sum + T[d][k]->child[ch]->lw;     // calculate sum of Le(s)
					//prod = prod * T[d][k]->child[ch]->pw;

				}

				//calculate weighted log-prob in two cases as explained in notes for numerical precision

				double delta = T[d][k]->le - sum + log2(1.0 * beta) - log2(1.0 - 1.0 * beta);
				if (delta < 30) {

					T[d][k]->lw = log2(1.0 - 1.0 * beta) + sum + log2(1.0 + 1.0 * pow(2.0, delta));
					//cout << pow(2, T[d][k]->lw) << endl;
				}
				else {
					T[d][k]->lw = log2(1.0 * beta) + T[d][k]->le + log2(1.0 * exp(1)) * (pow(2.0, -1.0 * delta) - pow(2.0, -2.0 * delta - 1));
					//cout << pow(2, T[d][k]->lw) << endl;
				}
				//T[d][k]->pw = beta * T[d][k]->p + (1.0 - 1.0*beta)*prod;
			}
		}
	}

	cout << "Mean marginal likelihood is " << pow(2.0, T[0][0]->lw) << endl;
	return pow(2.0, T[0][0]->lw);              //output value of weighted prob Pw at root
}

long double rma3(tree& T) {                   // algorithm takes improper tree finds mean marg lik and makes tree proper

	for (int d = D; d > -1; d--) {           // loop over levels

		//int length = T[d].size();           // nodes of each level might change when making the tree proper, but this affects previous level

		for (int k = 0; k < T[d].size(); k++) {  // loop over initially existing nodes of each level

			if (d == D) {                   // if at max depth, node is a leaf
				T[d][k]->lw = T[d][k]->le;
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (int ch = 0; ch < m; ch++) {

					if (T[d][k]->child[ch] != NULL) {           // if child #ch exists
						sum = sum + T[d][k]->child[ch]->lw;     // calculate sum of Le(s)
						//prod = prod * T[d][k]->child[ch]->pw;
					}

				}

				//calculate weighted log-prob in two cases as explained in notes for numerical precision

				long double delta = T[d][k]->le - sum + log2(beta) - log2(1.0 - beta);
				if (delta < 30) {

					T[d][k]->lw = log2(1.0 - beta) + sum + log2(1.0 + pow(2.0, delta));
					//cout << pow(2, T[d][k]->lw) << endl;
				}
				else {
					T[d][k]->lw = log2(beta) + T[d][k]->le + log2(exp(1)) * (pow(2.0, -delta) - pow(2.0, -2.0 * delta - 1));
					//cout << pow(2, T[d][k]->lw) << endl;
				}
				//T[d][k]->pw = beta * T[d][k]->p + (1.0 - 1.0*beta)*prod;
			}
		}
	}

	cout << "Mean marginal likelihood is " << pow(2.0, T[0][0]->lw) << " and has log " << T[0][0]->lw << endl;
	return T[0][0]->lw;              //output value of weighted prob Pw at root
}

long double mapt2(tree& T) {                   // algorithm takes improper tree finds maximal probability at root and makes tree proper, then finds MAPT
	// lw here is the maximal probablity, not the weighted one

	//First forward pass (leaves to root) to calculate maximal probabilities Pm at every node

	if (D == 0) { // if iid data
		//cout << " Pm,root is " << pow(2.0, T[0][0]->le) << endl;
		return pow(2.0, T[0][0]->le);              //output value of weighted prob Pw at root
	}

	for (int d = D; d > -1; d--) {           // loop over levels

		//int length = T[d].size();           // nodes of each level might change when making the tree proper, but this affects previous level

		for (int k = 0; k < T[d].size(); k++) {  // loop over initially existing nodes of each level

			if (d == D) {                   // if at max depth, node is a leaf
				T[d][k]->lw = T[d][k]->le;
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (short ch = 0; ch < m; ch++) {

					if (T[d][k]->child[ch] == NULL) {           // if child #ch does not exist, create it to make tree proper

						node* init = new node;
						T[d][k]->child[ch] = init;             // connect child with parent node
						T[d + 1].push_back(init);              // store at appropriate tree depth
						init->leaf = 1;                        // denote it leaf

						if (d < D - 1) {
							init->lw = log2(beta);            // set maximal prob for leaf at depth < D
						}

						//init->s = T[d][k]->s;                 // if contexts used
						//init->s.push_back(ch);                // if contexts used

					}

					sum = sum + T[d][k]->child[ch]->lw;       // sum of log-probs at children
					//cout << sum << endl;
				}

				// calculate maximal log-prob as explained in notes

				if (log2(1.0 - 1.0 * beta) + sum > log2(beta) + T[d][k]->le) { // maximum achieved by children term

					T[d][k]->lw = log2(1.0 - 1.0 * beta) + sum;                // set max prob of node
					//cout << pow(2, T[d][k]->lw) << endl;

				}

				else {                                                        // maximum achived by curent node

					T[d][k]->lw = log2(beta) + T[d][k]->le;                  // set max prob of node
					//cout << pow(2, T[d][k]->lw) << endl;

				}


			}
		}
	}

	// Then backward pass (root to leaves), to prune tree and destroy the required nodes
	// Use a[0] =-1 to mark nodes that need to be pruned

	for (int d = 0; d < D + 1; d++) { // root to leaves now

		int length = T[d].size();

		for (int k = 0; k < T[d].size(); k++) {

			if ((T[d][k]->lw == log2(beta) + T[d][k]->le) || (T[d][k]->a[0] == -1)) { // in this case make node leaf and prune all children

				if (T[d][k]->leaf == 0) {                                    // if node is not a leaf
					T[d][k]->leaf = 1;                                       // set node to be a leaf
					//cout << "prune children of " << d << k << endl;

					for (short ch = 0; ch < m; ch++) {

						T[d][k]->child[ch]->a[0] = -1;                         // mark children to be destructed later
						T[d][k]->child[ch] = 0;                            // destruct connections with children
					}
				}

			}

			if (T[d][k]->a[0] == -1) {                                   // node was marked to be destructed

				//cout << "delete " << d << k << endl;
				delete T[d][k];                                      // destruct node
				T[d].erase(T[d].begin() + k);                    // destruct the pointer of the node from the tree
				k--;
			}

		}
	}

	//cout << " Pm,root is " << pow(2.0, T[0][0]->lw) << endl;
	return pow(2.0, T[0][0]->lw);              //output value of weighted prob Pw at root
}

long double mapt(tree& T) {                   // algorithm takes improper tree finds maximal probability at root and makes tree proper, then finds MAPT
	// lw here is the maximal probablity, not the weighted one

	// First forward pass (leaves to root) to calculate maximal probabilities Pm at every node

	if (D == 0) { // if iid data
		//cout << " Pm,root is " << pow(2.0, T[0][0]->le) << endl;
		return  T[0][0]->le;              //output value of max prob at root
	}

	for (int d = D; d > -1; d--) {           // loop over levels

		// int length = T[d].size();           // nodes of each level might change when making the tree proper, but this affects previous level

		for (int k = 0; k < T[d].size(); k++) {  // loop over initially existing nodes of each level

			if (d == D) {                   // if at max depth, node is a leaf (if and only if for improper tree)
				T[d][k]->lw = T[d][k]->le;
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (short ch = 0; ch < m; ch++) {

					if (T[d][k]->child[ch] == NULL) {           // if child #ch does not exist, it is equivalent with

						if (d < D - 1) {
							sum = sum + log2(beta);
						}

					}

					else {                                        // if child ch exists

						sum = sum + T[d][k]->child[ch]->lw;       // sum of log-probs at children

					}

				}

				// calculate maximal log-prob as explained in notes

				//cout << "current: " << log2(beta) + T[d][k]->le << " children: " << log2(1.0 - 1.0*beta) + sum << endl;

				if (log2(1.0 - 1.0 * beta) + sum > log2(beta) + T[d][k]->le) { // maximum achieved by children term

					T[d][k]->lw = log2(1.0 - 1.0 * beta) + sum;                // set max prob of node
					//cout << pow(2, T[d][k]->lw) << endl;

				}

				else {                                                        // maximum achived by curent node

					T[d][k]->lw = log2(beta) + T[d][k]->le;                  // set max prob of node and mark to be pruned
					T[d][k]->leaf = 1;

					for (short ch = 0; ch < m; ch++) {   // for child # ch of each node


						//cout << "prune child " << ch << " of node " << d << k << endl;

						if (T[d][k]->child[ch] != NULL) {                       // if child exists

							T[d][k]->child[ch]->a[0] = -1;                     // mark child to be destructed
							T[d][k]->child[ch] = 0;                            // destruct connection with child

						}

					}
				}


			}
		}
	}

	cout << endl << " end of mapt forward pass" << endl;
	// Then backward pass (root to leaves), to prune tree and destroy the required nodes
	// Use M =-1 to mark nodes that need to be pruned

	int length[D + 1] = { 0 };
	for (int d = 0; d < D + 1; d++) {
		length[d] = T[d].size();
	}

	for (int d = 0; d < D + 1; d++) { // root to leaves now



		for (int k = 0; k < length[d]; k++) {

			if ((T[d][k]->a[0] == -1)) {  // node was marked to be deleted

				for (short ch = 0; ch < m; ch++) {   // for child # ch of each node

					if (T[d][k]->child[ch] != NULL) {                       // if child exists

						T[d][k]->child[ch]->a[0] = -1;                     // mark child to be destructed

					}
				}

				delete T[d][k];                                   // destruct node
				T[d].erase(T[d].begin() + k);                    // destruct the pointer of the node from the tree
				k--;
				length[d]--;
				//T[d].shrink_to_fit();                          // releases memory but takes much more time
			}

			else {

				if (T[d][k]->leaf == 0) {               // if child not a leaf and not deleted

					for (short ch = 0; ch < m; ch++) {

						if (T[d][k]->child[ch] == NULL) {


							node* init = new node;                // insert child to make tree proper
							T[d][k]->child[ch] = init;             // connect child with parent node
							T[d + 1].push_back(init);              // store at appropriate tree depth
							init->leaf = 1;                        // denote it leaf

							if (d < D - 1) {
								init->lw = log2(beta);            // set maximal prob for leaf at depth < D, if leaf is at depth D then logP=0;
							}

							//init->s = T[d][k]->s;                 // if contexts used
							//init->s.push_back(ch);                // if contexts used
						}
					}
				}
			}



		}
	}

	cout << " Pm,root is " << pow(2.0, T[0][0]->lw) << " and has log " << T[0][0]->lw << endl;
	return T[0][0]->lw;              // output value of weighted prob Pw at root
}

long double mapt4(tree& T) {                   // algorithm takes improper tree finds maximal probability at root and makes tree proper, then finds MAPT
	// lw here is the maximal probablity, not the weighted one

	// First forward pass (leaves to root) to calculate maximal probabilities Pm at every node

	if (D == 0) { // if iid data
		//cout << " Pm,root is " << pow(2.0, T[0][0]->le) << endl;
		return  T[0][0]->le;              //output value of max prob at root
	}

	for (int d = D; d > -1; d--) {           // loop over levels

		// int length = T[d].size();           // nodes of each level might change when making the tree proper, but this affects previous level

		for (int k = 0; k < T[d].size(); k++) {  // loop over initially existing nodes of each level

			if (d == D) {                   // if at max depth, node is a leaf (if and only if for improper tree)
				T[d][k]->lw = T[d][k]->le;
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (short ch = 0; ch < m; ch++) {

					if (T[d][k]->child[ch] == NULL) {           // if child #ch does not exist, it is equivalent with

						if (d < D - 1) {
							sum = sum + log2(beta);
						}

					}

					else {                                        // if child ch exists

						sum = sum + T[d][k]->child[ch]->lw;       // sum of log-probs at children

					}

				}

				// calculate maximal log-prob as explained in notes

				//cout << "current: " << log2(beta) + T[d][k]->le << " children: " << log2(1.0 - 1.0*beta) + sum << endl;

				if (log2(1.0 - 1.0 * beta) + sum > log2(beta) + T[d][k]->le) { // maximum achieved by children term

					T[d][k]->lw = log2(1.0 - 1.0 * beta) + sum;                // set max prob of node
					//cout << pow(2, T[d][k]->lw) << endl;

				}

				else {                                                        // maximum achived by curent node

					T[d][k]->lw = log2(beta) + T[d][k]->le;                  // set max prob of node and mark to be pruned
					T[d][k]->leaf = 1;

					for (short ch = 0; ch < m; ch++) {   // for child # ch of each node


						//cout << "prune child " << ch << " of node " << d << k << endl;

						if (T[d][k]->child[ch] != NULL) {                       // if child exists

							T[d][k]->child[ch]->a[0] = -1;                     // mark child to be destructed
							T[d][k]->child[ch] = 0;                            // destruct connection with child

						}

					}
				}


			}
		}
	}

	cout << endl << " end of mapt forward pass" << endl;
	// Then backward pass (root to leaves), to prune tree and destroy the required nodes
	// Use M =-1 to mark nodes that need to be pruned

	int length[D + 1] = { 0 };
	for (int d = 0; d < D + 1; d++) {
		length[d] = T[d].size();
	}



	for (int d = 0; d < D + 1; d++) { // root to leaves now

		int check = 0;

		for (int k = 0; k < length[d]; k++) {

			if ((T[d][k]->a[0] == -1)) {  // node was marked to be deleted

				for (short ch = 0; ch < m; ch++) {   // for child # ch of each node

					if (T[d][k]->child[ch] != NULL) {                       // if child exists

						T[d][k]->child[ch]->a[0] = -1;                     // mark child to be destructed

					}
				}

				//delete T[d][k];                                   // destruct node
				//T[d].erase(T[d].begin() + k);                    // destruct the pointer of the node from the tree
				//k--;
				//length[d]--;
				//T[d].shrink_to_fit();                          // releases memory but takes much more time
			}

			else {

				if (T[d][k]->leaf == 0) {               // if child not a leaf and not deleted

					for (short ch = 0; ch < m; ch++) {

						if (T[d][k]->child[ch] == NULL) {


							node* init = new node;                // insert child to make tree proper
							T[d][k]->child[ch] = init;             // connect child with parent node
							T[d + 1].push_back(init);              // store at appropriate tree depth
							init->leaf = 1;                        // denote it leaf

							if (d < D - 1) {
								init->lw = log2(beta);            // set maximal prob for leaf at depth < D, if leaf is at depth D then logP=0;
							}

							//init->s = T[d][k]->s;                 // if contexts used
							//init->s.push_back(ch);                // if contexts used
						}
					}
				}
			}

			if (T[d][k]->a[0] == -1) { //node doesn't exist in reality
				check++;
			}
		}
		if (check == length[d]) {
			cout << " Pm,root is " << pow(2.0, T[0][0]->lw) << " and has log " << T[0][0]->lw << endl;
			return T[0][0]->lw;              // output value of weighted prob Pw at root
		}
	}

	cout << " Pm,root is " << pow(2.0, T[0][0]->lw) << " and has log " << T[0][0]->lw << endl;
	return T[0][0]->lw;              // output value of weighted prob Pw at root
}

long double mapt3(tree& T) {                   // algorithm takes !!!proper!!! tree finds maximal probability at root and MAPT
	// lw here is the maximal probablity, not the weighted one

	//First forward pass (leaves to root) to calculate maximal probabilities Pm at every node

	if (D == 0) { // if iid data
		//cout << " Pm,root is " << pow(2.0, T[0][0]->le) << endl;
		return pow(2.0, T[0][0]->le);              //output value of weighted prob Pw at root
	}

	for (int d = D; d > -1; d--) {           // loop over levels

		//int length = T[d].size();           // nodes of each level might change when making the tree proper, but this affects previous level

		for (int k = 0; k < T[d].size(); k++) {  // loop over initially existing nodes of each level

			if (T[d][k]->leaf == 1) {                   // if node is a leaf

				if (d == D) {
					T[d][k]->lw = T[d][k]->le;
				}

				else {
					T[d][k]->lw = log2(beta);
				}
				//T[d][k]->pw = T[d][k]->p;
				//cout << pow(2, T[d][k]->lw) << endl;
			}
			else {                         // if node is not a leaf

				long double sum = 0;
				//long double prod = 1;
				for (short ch = 0; ch < m; ch++) {

					sum = sum + T[d][k]->child[ch]->lw;       // sum of log-probs at children
					//cout << sum << endl;
				}

				// calculate maximal log-prob as explained in notes

				if (log2(1.0 - 1.0 * beta) + sum > log2(beta) + T[d][k]->le) { // maximum achieved by children term

					T[d][k]->lw = log2(1.0 - 1.0 * beta) + sum;                // set max prob of node
					//cout << pow(2, T[d][k]->lw) << endl;

				}

				else {                                                        // maximum achived by curent node

					T[d][k]->lw = log2(beta) + T[d][k]->le;                  // set max prob of node
					//cout << pow(2, T[d][k]->lw) << endl;

				}


			}
		}
	}

	// Then backward pass (root to leaves), to prune tree and destroy the required nodes
	// Use M =-1 to mark nodes that need to be pruned

	for (int d = 0; d < D + 1; d++) { // root to leaves now

		//int length = T[d].size();

		for (int k = 0; k < T[d].size(); k++) {

			if ((T[d][k]->lw == log2(beta) + T[d][k]->le) || (T[d][k]->a[0] == -1)) { // in this case make node leaf and prune all children

				if (T[d][k]->leaf == 0) {                                    // if node is not a leaf
					T[d][k]->leaf = 1;                                       // set node to be a leaf
					//cout << "prune children of " << d << k << endl;

					for (short ch = 0; ch < m; ch++) {

						T[d][k]->child[ch]->a[0] = -1;                         // mark children to be destructed later
						T[d][k]->child[ch] = 0;                            // destruct connections with children
					}
				}

			}

			if (T[d][k]->a[0] == -1) {                                   // node was marked to be destructed

				//cout << "delete " << d << k << endl;
				delete T[d][k];                                      // destruct node
				T[d].erase(T[d].begin() + k);                    // destruct the pointer of the node from the tree
				k--;
			}

		}
	}

	cout << " Pm,root is " << pow(2.0, T[0][0]->lw) << " and has log " << T[0][0]->lw << endl;
	return  T[0][0]->lw;              //output value of weighted prob Pw at root
}

void rma_mapt(tree& T) {

	//tree * dentro = &T; // pointer for current tree

	long double pwl = rma3(T);

	cout << "end of rma" << endl;

	long double pml = mapt(T);

	cout << "end of mapt" << endl;

	label(T);

	int n_leaves = show_tree(T); //choose to show full tree

	n_leaves = show_leaves(T);  // choose to show leaves only


	long double prior = log2(pow(alpha, (n_leaves - 1.0)) * pow(beta, (n_leaves - T[D].size())));// log-prior


	cout << "prior is " << pow(2, prior) << " and has log " << prior << endl;

	long double posterior = pml - pwl; // log-posterior

	cout << "Posterior is " << pow(2, posterior) << " and has log " << posterior << endl;




}

void rma_mapt2(tree& T) {

	//tree * dentro = &T; // pointer for current tree

	long double pwl = rma3(T);

	cout << "end of rma" << endl;

	long double pml = mapt4(T);

	cout << "end of mapt" << endl;

	T = copytree(T);

	cout << "deleted nodes" << endl;

	label(T);

	int n_leaves = show_tree(T); //choose to show full tree

	n_leaves = show_leaves(T);  // choose to show leaves only


	long double prior = log2(pow(alpha, (n_leaves - 1.0)) * pow(beta, (n_leaves - T[D].size())));// log-prior


	cout << "prior is " << pow(2, prior) << " and has log " << prior << endl;

	long double posterior = pml - pwl; // log-posterior

	cout << "Posterior is " << pow(2, posterior) << " and has log " << posterior << endl;




}

vector<short> read(string s) { //could use char

	vector <short> xn = { 0 };
	xn.pop_back();

	ifstream fin;
	fin.open(s);

	char c = fin.get() - '0';

	while (fin.good()) {

		xn.push_back(c);
		c = fin.get() - '0';

	}

	fin.close();


	cout << "Size of Xn is " << xn.size() << endl;


	return xn;
}

vector<short> read2(string s) { //could use char, here for m > 9, in text file: each symbol in new line, after final symbol need one extra newline

	vector <short> xn = { 0 };
	xn.pop_back();

	ifstream fin;
	fin.open(s);

	char c[3] = { 0 };

	fin.getline(c, 3);


	while (fin.good()) {

		xn.push_back(atoi(c));
		fin.getline(c, 3);

	}

	fin.close();


	cout << "Size of Xn is " << xn.size() << endl;


	return xn;
}


void label(tree& T) { // takes as input proper tree with no contexts and writes their context in node.s

	for (int d = 0; d < D + 1; d++) {

		for (int k = 0; k < T[d].size(); k++) {
			//cout << d << k << endl;
			if (T[d][k]->leaf == 0) {

				for (short ch = 0; ch < m; ch++) {

					T[d][k]->child[ch]->s = T[d][k]->s;
					T[d][k]->child[ch]->s.push_back(ch);
				}


			}
		}

	}

}

void label2(tree& T) { // this version for mapt with "end of decoding at some level"

	for (int d = 0; d < D + 1; d++) {

		int check = 0;

		for (int k = 0; k < T[d].size(); k++) {

			if (T[d][k]->a[0] > -1) {

				if (T[d][k]->leaf == 0) {

					for (short ch = 0; ch < m; ch++) {

						T[d][k]->child[ch]->s = T[d][k]->s;
						T[d][k]->child[ch]->s.push_back(ch);
					}
				}
			}

			else { check++; }
		}

		if (check == T[d].size())
		{
			d = D + 5; //break out of loop
		}
	}
}

void deltree(tree& T) { // take output from mapt4 not deleting nodes because it takes time, this function deletes tha nodes to release memory

	for (int d = 0; d < D + 1; d++) {

		for (int k = 0; k < T[d].size(); k++) {

			if (T[d][k]->a[0] == -1) {

				delete T[d][k];
				// or delete[] T[d][k];

				T[d].erase(T[d].begin() + k);                    // destruct the pointer of the node from the tree
				k--;
			}
		}
	}


}


tree copytree(tree T) {   // takes tree from mapt4 wwhich doesn't delete nodes (only marks them so virtually they dont exist)
	// copies only non-marked nodes to another  tree, can use T=copytree(T) to release memory?
	// initialise tree of depth D
	node new_node;
	vector <node*> row(1);
	row[0] = &new_node;


	tree T2(D + 1, row);
	init_tree(T2);
	T2[0].pop_back();

	for (int d = 0; d < D + 1; d++) {

		int check = 0;

		for (int k = 0; k < T[d].size(); k++) {

			if (T[d][k]->a[0] > -1) {

				T2[d].push_back(T[d][k]);
			}

			else { check++; }
		}

		if (check == T[d].size()) {
			return T2;
		}
	}

	return T2;

}

void comb_int(int n, int r, vector <int> list) { // finds all combinations of integers in list, not used in code, 
	// was only used when writing code
	n--;
	vector<int> v;
	for (int i = 0; i <= r; ++i) {
		v.push_back(0);
	}
	while (true) {
		for (int i = 0; i < r; ++i) {                //vai um
			if (v[i] > n) {
				v[i + 1] += 1;
				for (int k = i; k >= 0; --k) {
					v[k] = v[i + 1];
				}
				//v[i] = v[i + 1];
			}
		}

		if (v[r] > 0) { break; }
		for (short i = 0; i < r; i++) {
			cout << list[v[i]];
		}
		cout << endl;
		v[0] += 1;
	}



}

void comb_initial(int n, int r, int d, vector <node*> init) { // wrong version to find combinations of children products
	// it doesn't give 12 and 21 beause they are the same
	n--;
	vector<int> v;
	for (int i = 0; i <= r; ++i) {
		v.push_back(0);
	}
	while (true) {
		for (int i = 0; i < r; ++i) {                //vai um
			if (v[i] > n) {
				v[i + 1] += 1;
				for (int k = i; k >= 0; --k) {
					v[k] = v[i + 1];
				}

			}
		}

		if (v[r] > 0) { break; }

		double sum = log2(1.0 - 1.0 * beta);

		for (short i = 0; i < r; i++) {

			sum = sum + init[d + 1]->lm[v[i]];
		}

		vector <short> ci; // equiv of c[i] in other algorithm

		for (short i = 0; i < r; i++) {
			ci.push_back(v[i] + 1);
		}

		vector <vector <short>> tempc;

		vector <double> temp;



		if (sum > init[d]->lm[init[d]->lm.size() - 1]) {

			int j = init[d]->lm.size() - 1;
			bool test = 0;

			while (j > 0) {

				if ((sum >= init[d]->lm[j]) && (sum <= init[d]->lm[j - 1])) {

					temp.push_back(sum);
					tempc.push_back(ci);

					short limit = 0;

					if (init[d]->lm.size() < k_max) {
						limit = 1;
					}

					for (int q = 0; q < init[d]->lm.size() - j - 1 + limit; q++) {
						temp.push_back(init[d]->lm[j + q]);
						tempc.push_back(init[d]->c[j + q]);
					}

					if (init[d]->lm.size() < k_max) {
						init[d]->lm.push_back(0);
						init[d]->c.push_back({ 0 });
					}

					for (int q = 0; q < temp.size(); q++) {

						init[d]->lm[j + q] = temp[q];
						init[d]->c[j + q] = tempc[q];
					}

					j = 1;
					test = 1;
				}

				j--;

			}

			if (test == 0) {

				temp.push_back(sum);
				tempc.push_back(ci);

				short limit = 0;

				if (init[d]->lm.size() < k_max) {
					limit = 1;
				}

				for (int q = 0; q < init[d]->lm.size() - 1 + limit; q++) {
					temp.push_back(init[d]->lm[q]);
					tempc.push_back(init[d]->c[q]);
				}

				if (init[d]->lm.size() < k_max) {
					init[d]->lm.push_back(0);
					init[d]->c.push_back({ 0 });
				}

				for (int q = 0; q < temp.size(); q++) {

					init[d]->lm[q] = temp[q];
					init[d]->c[q] = tempc[q];
				}

			}
		}

		else {

			if (init[d]->lm.size() < k_max) {
				init[d]->lm.push_back(sum);
				init[d]->c.push_back(ci);
			}
		}

		v[0] += 1;
	}



}

void preproc(vector <node*> init) {   // preprocessing needed for k-mapt, gives init[0] for node at d=1...init[D-1] for d=D
	// no need to include root here as it is always in the data (obviously)


	init[D - 1]->c.push_back(zeros);       // for d=D , c=0 and lm[0]=0 from construction 

	for (short d = D - 2; d > -1; d--) {

		init[d]->lm[0] = log2(beta);      // for smaller depth first add c=0 with p=logbeta
		init[d]->c.push_back(zeros);

		//comb_initial(init[d + 1]->lm.size(), m, d, init);
		comb_initial3(d, init);           // then find all combinations and keep the top k of them

	}

	//for (int d = 0; d < D; d++) {         // just prints the preprocessing c's, not needed actually
	//cout << "d = " << d << endl;
	//for (int j = 0; j < init[d]->lm.size(); j++)
	//{
	//cout << endl << init[d]->lm[j] << " and vector is ";
	//	for (int i = 0; i < m; i++) {
	//	cout << init[d]->c[j][i];
	//}
	//cout << endl;
	//}
	//}
}


vector<vector<double> > cart_product(const vector<vector<double>>& v) { // cartesian product of vectors
	vector<vector<double>> s = { {} };                                  // returns a matrix with all combinations
	for (auto& u : v) {
		vector<vector<double>> r;
		for (auto& x : s) {
			for (auto y : u) {
				r.push_back(x);
				r.back().push_back(y);
			}
		}
		s.swap(r);
	}
	return s;
}

vector<vector<short> > cart_product_int(const vector<vector<short>>& v) {   // finds cartesian product of vector of short ints
	vector<vector<short>> s = { {} };                                       // stores in matrix
	for (auto& u : v) {
		vector<vector<short>> r;
		for (auto& x : s) {
			for (auto y : u) {
				r.push_back(x);
				r.back().push_back(y);
			}
		}
		s.swap(r);
	}
	return s;
}

void comb(int d, int k, tree& T, vector <node*> init) {                   // finds combinations of children of node T[d][k]
	// computes the products for lm and keeps top k 
	vector<vector<double>> v;  // stores combinations of probabilities
	vector<vector<short>> c;  // stores respective indices


	for (short ch = 0; ch < m; ch++) {


		node* pointer;

		if (T[d][k]->child[ch] == NULL) {

			pointer = init[d];                         // if child does not exists (not occured) use preprocessing node

		}

		else {

			pointer = T[d][k]->child[ch];                 // else use children of node

		}

		v.push_back(pointer->lm);

		vector <short> temp;

		for (short j = 1; j < pointer->lm.size() + 1; j++) { // use cartesian products of {1,2,3}... to find position vectors

			temp.push_back(j);
		}

		c.push_back(temp);
	}

	vector<vector<double> > v2 = cart_product(v);          // cartesian products : all combinations
	c = cart_product_int(c);

	for (int i = 0; i < v2.size(); i++) {                  //loop combiantions, keep what is necessary

		double sum = log2(1.0 - beta);

		for (short j = 0; j < v2[i].size(); j++) {
			sum = sum + v2[i][j];                          // sum of log-lm= prod-lm
		}

		vector <vector <short>> tempc;                    // use temporal variables to keep sort approprately

		vector <double> temp;

		if (sum > T[d][k]->lm[T[d][k]->lm.size() - 1]) {  // if sum> min of ordered list lm
			// could also use >= ( need to change appropriately below)
			int j = T[d][k]->lm.size() - 1;
			bool test = 0;

			while (j > 0) {

				if ((sum > T[d][k]->lm[j]) && (sum <= T[d][k]->lm[j - 1])) {  // if > of prev and < next, inlude there
					// could also use other combinaitons of <=, >=
					// stick element below the first one which is equal to sum
					temp.push_back(sum);
					tempc.push_back(c[i]);

					short limit = 0;

					if (T[d][k]->lm.size() < k_max) {
						limit = 1;                       // if the size of lm<k then add without replacement
					}

					for (int q = 0; q < T[d][k]->lm.size() - j - 1 + limit; q++) { //set temp
						temp.push_back(T[d][k]->lm[j + q]);
						tempc.push_back(T[d][k]->c[j + q]);
					}

					if (T[d][k]->lm.size() < k_max) {   // if size<k then add to list (here initialise space)
						T[d][k]->lm.push_back(0);
						T[d][k]->c.push_back({ 0 });
					}

					for (int q = 0; q < temp.size(); q++) {

						T[d][k]->lm[j + q] = temp[q];    // use temp for assignments
						T[d][k]->c[j + q] = tempc[q];
					}

					j = 1;            // used to break out of loop when added thi combination
					test = 1;         // test used to add element at the top of the list
				}

				j--;

			}

			if (test == 0) {         // if sum> all lm then add it at the top of the list
				// code identical with before at index 0

				temp.push_back(sum);
				tempc.push_back(c[i]);

				short limit = 0;

				if (T[d][k]->lm.size() < k_max) {
					limit = 1;
				}

				for (int q = 0; q < T[d][k]->lm.size() - 1 + limit; q++) {
					temp.push_back(T[d][k]->lm[q]);
					tempc.push_back(T[d][k]->c[q]);
				}

				if (T[d][k]->lm.size() < k_max) {
					T[d][k]->lm.push_back(0);
					T[d][k]->c.push_back({ 0 });
				}

				for (int q = 0; q < temp.size(); q++) {

					T[d][k]->lm[q] = temp[q];
					T[d][k]->c[q] = tempc[q];
				}

			}
		}

		else { // if element <= last element of list, only include it at the bottom if size<k

			if (T[d][k]->lm.size() < k_max) {
				T[d][k]->lm.push_back(sum);
				T[d][k]->c.push_back(c[i]);
			}
		}





	}


}

void comb_initial2(int d, vector <node*> init) {      // 2nd version for combination of initialisation. not used as well
	// trial version to choose more symmetric position vectors at ties
	// actually works for k<=5, but maybe not for more, currently not used
	// probablty not an important thing to optimize as gives same results with combinitial3

	vector<vector<double>> v;  // stores combinations of probabilities
	vector<vector<short>> c;  // stores respective indices


	for (short ch = 0; ch < m; ch++) {


		node* pointer;
		pointer = init[d + 1];


		v.push_back(pointer->lm);

		vector <short> temp;

		for (short j = 1; j < pointer->lm.size() + 1; j++) {

			temp.push_back(j);
		}

		c.push_back(temp);
	}

	vector<vector<double> > v2 = cart_product(v);
	c = cart_product_int(c);
	cout << v2.size() << c.size() << endl;
	for (int i = 0; i < v2.size(); i++) {

		double sum = log2(1.0 - beta);

		for (short j = 0; j < v2[i].size(); j++) {
			sum = sum + v2[i][j];
		}

		vector <vector <short>> tempc;

		vector <double> temp;

		if (sum >= init[d]->lm[init[d]->lm.size() - 1]) {

			int j = init[d]->lm.size() - 1;
			bool test = 0;
			int rep = 0;

			while (j > 0) {

				if (sum == init[d]->lm[j]) {
					rep++;
				}

				bool symmetry = 0;

				if ((sum == init[d]->lm[j]) && (sum < init[d]->lm[j - 1])) {

					if (difference(c[i]) < difference(init[d]->c[j])) {
						symmetry = 1;
					}

					else {

						if (j < init[d]->lm.size() - 1) {
							symmetry = 1;
							j++;
						}

						else {

							if (init[d]->lm.size() < k_max) {

								init[d]->lm.push_back(sum);
								init[d]->c.push_back(c[i]);

							}

							j = 0;
							test = 1;


						}

					}

				}

				if (((sum > init[d]->lm[j]) && (sum < init[d]->lm[j - 1]) || (symmetry == 1))) {

					temp.push_back(sum);
					tempc.push_back(c[i]);

					short limit = 0;

					if (init[d]->lm.size() < k_max) {
						limit = 1;
					}

					for (int q = 0; q < init[d]->lm.size() - j - 1 + limit; q++) {
						temp.push_back(init[d]->lm[j + q]);
						tempc.push_back(init[d]->c[j + q]);
					}

					if (init[d]->lm.size() < k_max) {
						init[d]->lm.push_back(0);
						init[d]->c.push_back({ 0 });
					}

					for (int q = 0; q < temp.size(); q++) {

						init[d]->lm[j + q] = temp[q];
						init[d]->c[j + q] = tempc[q];
					}

					j = 1;
					test = 1;
				}

				j--;

			}

			if (test == 0) {

				temp.push_back(sum);
				tempc.push_back(c[i]);

				short limit = 0;

				if (init[d]->lm.size() < k_max) {
					limit = 1;
				}

				for (int q = 0; q < init[d]->lm.size() - 1 + limit; q++) {
					temp.push_back(init[d]->lm[q]);
					tempc.push_back(init[d]->c[q]);
				}

				if (init[d]->lm.size() < k_max) {
					init[d]->lm.push_back(0);
					init[d]->c.push_back({ 0 });
				}

				for (int q = 0; q < temp.size(); q++) {

					init[d]->lm[q] = temp[q];
					init[d]->c[q] = tempc[q];
				}

			}
		}

		else {

			if (init[d]->lm.size() < k_max) {
				init[d]->lm.push_back(sum);
				init[d]->c.push_back(c[i]);
			}
		}





	}


}


void kmapt_forw(tree& T, vector <node*> init) { // forward pass of kmapt algorithm

	for (int d = D; d > -1; d--) {               // loop for leaves to root
		//cout << "size is " << T[d].size() << endl;
		for (int k = 0; k < T[d].size(); k++) {
			//cout << "d is " << d << " and k is " << k << endl;
			if (d == D) {

				T[d][k]->lm[0] = T[d][k]->le;    // at leaves set lm[0]=le with position vector c=0
				T[d][k]->c.push_back(zeros);

				//cout << "size is" << T[d][k]->lm.size() << endl;
				//cout << "prob is " << T[d][k]->lm[0];
				//cout << " and position vector is ";
				//for (short ch = 0; ch < m; ch++) {
				//cout << T[d][k]->c[0][ch];
				//}
				//cout << endl;

			}


			else {

				T[d][k]->lm[0] = log2(beta) + T[d][k]->le; // for d<D first add the c=0 lm=le combination
				T[d][k]->c.push_back(zeros);               // if sth is equal to that it will be stuck beloaw it
				// intuitevely keep c=0 higher to "reward" pruning at ties

				comb(d, k, T, init);                       // find combinations, short and keep top k of them in list

				//for (short p = 0; p < T[d][k]->lm.size(); p++) {
				//cout << "prob is " << T[d][k]->lm[p];
				//cout << " and position vector is ";
				//for (short ch = 0; ch < m; ch++) {
				//	cout << T[d][k]->c[p][ch];
				//}
				//cout << endl;
				//}
			}
		}
	}

	for (short p = 0; p < T[0][0]->lm.size(); p++) {   //output at root
		//cout << "prob is " << T[0][0]->lm[p] << endl;

	}
}

void kmapt_back(vector <node*> init, tree T, vector <tree>& trees) {  // backward loop of kmapt
	// builds top k trees using the lm and c's

	for (int i = 0; i < k_max; i++) { // loop over k-top trees

		//cout << "i is " << i << endl;


		if (T[0][0]->c[i][0] != 0) {  // if the c at the root is not zero, then don't prune there and add nodes at d=1

			node* temp = new node;  // create new node
			*temp = *T[0][0];        // initialise from T (so have children)
			trees[i][0][0] = temp;   // add to new tree

			for (int ch = 0; ch < m; ch++) { // always add m children to have proper tree output

				if (trees[i][0][0]->child[ch] != NULL) { // if child exists, initialise new node like the child and
					node* temp2 = new node;             // add it to the tree at depth 1
					*temp2 = *trees[i][0][0]->child[ch];
					trees[i][1].push_back(temp2);
					trees[i][0][0]->child[ch] = temp2; // connect it to the root of the new tree
				}

				else {  // if child doesn't exist, initialise new node from preprocessing step anda add to tree, connect to root
					node* newnode = new node;
					*newnode = *init[0];
					trees[i][1].push_back(newnode);
					trees[i][0][0]->child[ch] = newnode;
				}
			}

			for (int d = 0; d < D - 1; d++) { // after adding the m nodes at d=1, check about pruning iteratively
				//cout << "d is " << d << endl;
				for (int k = 0; k < trees[i][d].size(); k++) {
					//cout << "k is " << k << endl;
					if (trees[i][d][k]->leaf == 0) {   // if node is not a leaf

						for (int j = 0; j < m; j++) {  // for all its children (all of thm exist) so next line probably not necessary
							//cout << "j is " << j << endl;
							if (trees[i][d][k]->child[j]->leaf == 0) { // probably not necessary

								int index = 0;
								if (d == 0) { index = i; }
								short t = trees[i][d][k]->c[index][j] - 1; // index t as in notes (-1 because of c++ indexing from 0 and not from 1)

								//cout << "t is " << t << endl;
								//cout << trees[i][d][k]->child[j]->c[trees[i][d][k]->c[i][j] - 1][0] << endl;
								//bool prune = 0;

								//for (int l = 0; l < trees[i][d][k]->child[j]->lm.size(); l++) {

								//if ((trees[i][d][k]->child[j]->lm[l] == trees[i][d][k]->child[j]->lm[t]) && (trees[i][d][k]->child[j]->c[l][0] == 0)) {
								//	prune = 1;
								//}
								//}

								// check if appropriate c is not 0 then no pruning and adding all children

								if (trees[i][d][k]->child[j]->c[t][0] != 0) {   // or if trees[i][d][k]->child[j]->c[t][0] != 0 or prune==0
									//cout << 10101010110 << endl;
									//cout << trees[i][d][k]->child[j]->c.size() << endl;

									// IF STH WRONG COMMENT THIS IF OUT, this was added for error when D is small
									//this if says is trees[i][d][k]->child[j]->c[i] EXISTS then...
									trees[i][d][k]->child[j]->c[0] = trees[i][d][k]->child[j]->c[t]; // after node examined and children added, take t of next step from examined node
									// (node will not be examined again so this is ok)

									//else {  //OR UNCOMMENT THIS, problem where D small or k large, where trees[i][d][k]->child[j]->c[i] might not exist

									//	trees[i][d][k]->child[j]->c.push_back(trees[i][d][k]->child[j]->c[t]);
									//}

									for (int ch = 0; ch < m; ch++) {

										if (trees[i][d][k]->child[j]->child[ch] != NULL) {  // if child exists 

											node* temp3 = new node;                        // create new child
											*temp3 = *trees[i][d][k]->child[j]->child[ch];  // initialise from tree T
											trees[i][d + 2].push_back(temp3);               // add it to tree
											trees[i][d][k]->child[j]->child[ch] = temp3;    // connect it to parent
										}

										else { //if child doesn't exist initialise from preprocessing and then as before

											node* newnode2 = new node;
											*newnode2 = *init[d + 1];
											if (d == D - 2) {
												newnode2->leaf = 1; // the adding tree at depth D, so mark it a leaf
											}
											trees[i][d + 2].push_back(newnode2);
											trees[i][d][k]->child[j]->child[ch] = newnode2;
										}
									}
								}

								else { // if pruning at node 

									trees[i][d][k]->child[j]->leaf = 1; // mark it to be a leaf and make all children pointer =0 so they don't exist
									for (int ch = 0; ch < m; ch++) { trees[i][d][k]->child[j]->child[ch] = 0; }

								}
							}
						}
					}
				}
			}
		}

		else { // else if T[0][0] c[i]=0 then keep only root node

			node* newnode3 = new node;
			*newnode3 = *T[0][0];
			newnode3->leaf = 1;
			for (int ch = 0; ch < m; ch++) { newnode3->child[ch] = 0; }
			trees[i][0][0] = newnode3;
		}




	}




}

void kmapt(tree& T, vector <tree>& trees, vector <node*> init, vector<double>& odds) { // call all kmapt functions together

	long double pwl = rma3(T);

	cout << "end of rma" << endl;

	kmapt_forw(T, init);

	cout << " end of forward pass" << endl;

	kmapt_back(init, T, trees);

	cout << " end of backward pass" << endl;


	for (int i = 0; i < k_max; i++) {

		label(trees[i]);

		//int n_leaves = show_tree(trees[i]); //choose to show full tree

		int n_leaves = show_leaves(trees[i]);  // choose to show leaves only


		long double prior = log2(pow(alpha, (n_leaves - 1.0)) * pow(beta, (n_leaves - trees[i][D].size())));// log-prior


		cout << "prior is " << pow(2, prior) << " and has log " << prior << endl;

		cout << " Pm,root is " << pow(2.0, T[0][0]->lm[i]) << " and has log " << T[0][0]->lm[i] << endl;

		long double pml = T[0][0]->lm[i];

		long double posterior = pml - pwl; // log-posterior

		cout << "Posterior is " << pow(2, posterior) << " and has log " << posterior << endl;

		odds[i] = pow(2, T[0][0]->lm[0] - pml);

		cout << "Posterior odds = " << pow(2, T[0][0]->lm[0] - pml) << endl;
	}


}

int difference(vector <short> c) { // find distance between context adj symbols, used for choosing symmetric c's at ties
	// not used and probably not ok in general (have to use odd-even cases to mimprove)
	int sum = 0;

	for (int i = 1; i < c.size(); i++) {

		sum = sum + abs(c[i] - c[i - 1]);
	}

	return sum;
}

void comb_initial3(int d, vector <node*> init) {    // finds combinations from preprocessing stage, which is currently used
	// actually similar code with combinitial2, but used init[d] as current node	
	// and init[d+1] for children of node, see there for more comments

	vector<vector<double>> v;  // stores combinations of probabilities
	vector<vector<short>> c;  // stores respective indices


	for (short ch = 0; ch < m; ch++) {


		node* pointer;
		pointer = init[d + 1];


		v.push_back(pointer->lm);

		vector <short> temp;

		for (short j = 1; j < pointer->lm.size() + 1; j++) {

			temp.push_back(j);
		}

		c.push_back(temp);
	}

	vector<vector<double> > v2 = cart_product(v);
	c = cart_product_int(c);

	for (int i = 0; i < v2.size(); i++) {

		double sum = log2(1.0 - beta);

		for (short j = 0; j < v2[i].size(); j++) {
			sum = sum + v2[i][j];
		}

		vector <vector <short>> tempc;

		vector <double> temp;

		if (sum > init[d]->lm[init[d]->lm.size() - 1]) { // care taken here wih using > or >=

			int j = init[d]->lm.size() - 1;
			bool test = 0;

			while (j > 0) {

				if ((sum > init[d]->lm[j]) && (sum <= init[d]->lm[j - 1])) {// care taken here wih using > or >=, < or <=
					// see comemnts on comb 2 for more details
					temp.push_back(sum);
					tempc.push_back(c[i]);

					short limit = 0;

					if (init[d]->lm.size() < k_max) {
						limit = 1;
					}

					for (int q = 0; q < init[d]->lm.size() - j - 1 + limit; q++) {
						temp.push_back(init[d]->lm[j + q]);
						tempc.push_back(init[d]->c[j + q]);
					}

					if (init[d]->lm.size() < k_max) {
						init[d]->lm.push_back(0);
						init[d]->c.push_back({ 0 });
					}

					for (int q = 0; q < temp.size(); q++) {

						init[d]->lm[j + q] = temp[q];
						init[d]->c[j + q] = tempc[q];
					}

					j = 1;
					test = 1;
				}

				j--;

			}

			if (test == 0) {

				temp.push_back(sum);
				tempc.push_back(c[i]);

				short limit = 0;

				if (init[d]->lm.size() < k_max) {
					limit = 1;
				}

				for (int q = 0; q < init[d]->lm.size() - 1 + limit; q++) {
					temp.push_back(init[d]->lm[q]);
					tempc.push_back(init[d]->c[q]);
				}

				if (init[d]->lm.size() < k_max) {
					init[d]->lm.push_back(0);
					init[d]->c.push_back({ 0 });
				}

				for (int q = 0; q < temp.size(); q++) {

					init[d]->lm[q] = temp[q];
					init[d]->c[q] = tempc[q];
				}

			}
		}

		else {

			if (init[d]->lm.size() < k_max) {
				init[d]->lm.push_back(sum);
				init[d]->c.push_back(c[i]);
			}
		}





	}


}

long double mle(tree T) {  //finds mle of proper tree T 

	long double sum = 0;

	for (int d = 0; d < D + 1; d++) {

		for (int k = 0; k < T[d].size(); k++) { // loop over all leaves of the tree

			if (T[d][k]->leaf == 1) {

				int M = 0;

				for (int j = 0; j < m; j++) {
					M = M + T[d][k]->a[j];
				}

				for (int j = 0; j < m; j++) {

					if (T[d][k]->a[j] != 0) {

						sum = sum + 1.0 * T[d][k]->a[j] * log(1.0 * T[d][k]->a[j] / M);
					}

				}

			}
		}
	}
	cout << "Log ML is " << sum << endl;
	return sum;
}

long double mlevlmc(tree T) {  //finds mle of a general tree tree T 

	long double sum = 0;

	for (int d = 0; d < D + 1; d++) {

		for (int k = 0; k < T[d].size(); k++) { // loop over all leaves of the tree

			if (T[d][k]->leaf == 1) { // node is a leaf, so has no children

				int M = 0;

				for (int j = 0; j < m; j++) {
					M = M + T[d][k]->a[j];
				}

				for (int j = 0; j < m; j++) {

					if (T[d][k]->a[j] != 0) {

						sum = sum + 1.0 * T[d][k]->a[j] * log(1.0 * T[d][k]->a[j] / M);
					}

				}

			}

			else {//ie if node is not a leaf, then it has some children

				int check = 0;

				for (int ch = 0; ch < m; ch++) {
					if (T[d][k]->child[ch] != NULL) {
						check++;
					}
				}

				if (check < m) { //id it doesn't have all its m children

					int M = 0;

					for (int j = 0; j < m; j++) {
						M = M + T[d][k]->a[j];
					}

					int b[m] = { 0 };

					for (int j = 0; j < m; j++) {

						b[j] = T[d][k]->a[j];

						for (int ch = 0; ch < m; ch++) {

							if (T[d][k]->child[ch] != NULL) {

								b[j] = b[j] - T[d][k]->child[ch]->a[j];

							}
						}

					}
					M = 0;
					for (int j = 0; j < m; j++) {
						M = M + b[j];
					}

					for (int j = 0; j < m; j++) {

						if (b[j] != 0) {

							sum = sum + 1.0 * b[j] * log(1.0 * b[j] / M);
						}

					}



				}
			}
		}
	}
	cout << "Log ML is " << sum << endl;
	return sum;
}

long double mle2(tree T) {  //finds mle of improper tree T of max depth D 

	long double sum = 0;



	for (int k = 0; k < T[D].size(); k++) { // loop over all leaves of the tree


		int M = 0;

		for (int j = 0; j < m; j++) {
			M = M + T[D][k]->a[j];
		}

		for (int j = 0; j < m; j++) {

			if (T[D][k]->a[j] != 0) {

				sum = sum + 1.0 * T[D][k]->a[j] * log(1.0 * T[D][k]->a[j] / M);
			}

		}


	}

	cout << "Log ML is " << sum << endl;
	return sum;
}

void counts(vector <double> xn, tree& T) { // finds counts of proper tree T in sequence xn

	for (int d = 0; d < D + 1; d++) {  // first reset all counts to zero

		for (int k = 0; k < T[d].size(); k++) {

			for (int ch = 0; ch < m; ch++) {

				T[d][k]->a[ch] = 0;
			}
		}
	}

	int max;
	for (int i = 0; i < D + 1; i++) {
		if (T[i].size() > 0) {
			max = i;
		}
	}
	max = D;  // use D or max depth for initial value, comment this line out to use max depth

	// update for each sequence symbol
	for (int i = max; i < xn.size(); i++) {


		short s = xn[i];          // current symbol
		vector <short> ct(D);     // current context

		//cout << endl << "symbol " << i << ", with context ";

		for (int j = 0; j < D; j++) {

			ct[j] = xn[i - j - 1];  // sets context
			//cout << ct[j];   // prints context
		}
		//cout << endl;

		node* temp = T[0][0];
		T[0][0]->a[s]++;

		for (int j = 0; j < D; j++) {

			if (temp->leaf == 0) {

				temp = temp->child[ct[j]];   //proper tree so if not a leaf all children exist
				temp->a[s]++;
			}

			else {
				j = D + 5;
			}
		}





	}




}

void counts2(vector <double> xn, tree& T) { // finds counts of a general, maybe improper tree T, in sequence xn

	for (int d = 0; d < D + 1; d++) {  // first reset all counts to zero

		for (int k = 0; k < T[d].size(); k++) {

			for (int ch = 0; ch < m; ch++) {

				T[d][k]->a[ch] = 0;
			}
		}
	}

	int max;
	for (int i = 0; i < D + 1; i++) {
		if (T[i].size() > 0) {
			max = i;
		}
	}
	max = D;  // use D or max depth for initial value, comment this line out to use max depth
	// set max=1 for counts like vlmc in R
	// update for each sequence symbol
	for (int i = D; i < xn.size(); i++) {


		short s = xn[i];          // current symbol
		vector <short> ct(D);     // current context

		//cout << endl << "symbol " << i << ", with context ";

		for (int j = 0; j < D; j++) {

			ct[j] = xn[i - j - 1];  // sets context
			//cout << ct[j];   // prints context
		}
		//cout << endl;

		node* temp = T[0][0];
		T[0][0]->a[s]++;

		for (int j = 0; j < D; j++) {

			if (temp->leaf == 0) {

				if (temp->child[ct[j]] != NULL) {
					temp = temp->child[ct[j]];   //improper tree so if not a leaf a child might not exist exist
					temp->a[s]++;
				}
				else {
					j = D + 5;                  //if child doesn't exist, distribution is at internal node
				}
			}

			else {
				j = D + 5;
			}
		}





	}




}

long double bic(tree& T, vector <double> xn) { // find bic of proper tree T, sequence xn

	int n_leaves = show_leaves(T);

	counts(xn, T);

	long double ml = mle(T);

	long double bic = -2 * ml + (n_leaves * (m - 1)) * log(xn.size() - D);
	long double aic = -2 * ml + (n_leaves * (m - 1)) * 2;

	cout << "BIC is " << bic << endl;
	cout << "AIC is " << aic << endl;

	return bic;


}

long double bicvlmc(tree& T, vector <double> xn) { // find bic of general tree T, sequence xn

	int n_leaves = no_leaves_vlmc(T);
	show_tree(T);
	counts2(xn, T);

	long double ml = mlevlmc(T);

	long double bic = -2 * ml + (n_leaves * (m - 1)) * log(xn.size() - D);
	long double aic = -2 * ml + (n_leaves * (m - 1)) * 2;

	cout << "BIC is " << bic << endl;
	cout << "AIC is " << aic << endl;

	return bic;


}



long double logloss(vector <double> xn, int T) { // might need to CHECK all these for continuous case

	node new_node;
	vector <node*> row(1);
	row[0] = &new_node;


	tree Tr(D + 1, row);
	init_tree(Tr);

	vector <double> yn;

	for (int i = 0; i < xn.size() - T; i++) {
		yn.push_back(xn[i]);
	}

	//cout << size(yn) << endl;

	build_tree(Tr, yn);

	long double r1 = rma3(Tr);



	for (int i = xn.size() - T; i < xn.size(); i++) {
		//cout << xn[i] << endl; //prints sequence

		double s = xn[i];          // current symbol

		vector<double> x_tilde;     //continuous context needed for sums
		x_tilde.push_back(1.0);
		for (int j = 1; j < ar_p + 1; j++) {
			x_tilde.push_back(xn[i - j]);
		}

		vector <short> ct(D);     // current context

		//cout << endl << "symbol " << i << ", with context ";

		for (int j = 0; j < D; j++) { //here use QUANTISED version for disrete context

			//ct[j] = xn[i - j - 1];  // sets context
			//cout << ct[j];   // prints context
			if (xn[i - j] > xn[i - j - 1]) {
				ct[j] = 1;
			}
			else { ct[j] = 0; }
		}
		//cout << endl;

		update(Tr, s, ct, x_tilde);

	}

	cout << "Extra symbols were added" << endl;

	long double r2 = rma3(Tr);

	long double log_loss = (r2 - r1) / (1.0 * T);

	cout << "Log-loss is " << log_loss << endl;

	return log_loss;
}

tree2 construct(tree T) { // takes tree and find info required to construct tree2

	tree2 Tout;
	Tout.t = T;

	vector <node*> leaves;
	vector <node*> pl;
	int maxdepth = 0;


	for (int u = 0; u < D + 1; u++) {
		for (int v = 0; v < T[u].size(); v++)
		{

			if (T[u][v]->leaf == 1) {

				if (u < D) {
					leaves.push_back(T[u][v]);
				}
			}

			else {

				int c = 0;
				for (int ch = 0; ch < m; ch++) {
					if (T[u][v]->child[ch]->leaf == 1) {
						c++;
					}
				}

				if (c == m) {
					pl.push_back(T[u][v]);
				}

			}

			maxdepth = u;
		}
	}

	Tout.leaves = leaves;
	Tout.pl = pl;
	Tout.d = maxdepth;


	return Tout;
}

void makeproper(tree& T) { // takes improper tree and makes it proper



	for (int d = 0; d < D + 1; d++) {
		for (int k = 0; k < T[d].size(); k++) {

			if (T[d][k]->leaf == 0) {

				for (int ch = 0; ch < m; ch++) {

					if (T[d][k]->child[ch] == NULL) {

						node* init = new node;
						T[d][k]->child[ch] = init;             // connect child with parent node
						T[d + 1].push_back(init);              // store at appropriate tree depth
						init->leaf = 1;                        // denote it leaf
						init->s = T[d][k]->s;                  // if contexts used
						init->s.push_back(ch);


					}
				}
			}
		}
	}

}
void mcmc_jump(double p, map< vector < vector <short>>, tree2* >& mcmc_trees, vector <tree2> Tstar, short T0, map < vector <short>, node*> dict, int N, vector <double> odds) {


	vector < vector <short>> id;

	for (int i = 0; i < k_max; i++) {
		id = treeid(Tstar[i]);
		mcmc_trees[id] = new tree2;  //add top k trees in mcmc vector
		*mcmc_trees[id] = Tstar[i];
	}

	mcmc_trees[treeid(Tstar[T0])]->f = 1; //T0 just gives the integer index of the initial tree (use one of top k in this version)
	tree2 Tc = construct(copy(Tstar[T0].t)); // current tree in mcmc T

	gibbs_step(Tc, dict);


	bernoulli_distribution jump_tree(p);

	int n = 0;
	//default_random_engine generator2(431); check this
	vector <long double> h; //entropy rate 

	for (int i = 0; i < N; i++) {

		//cout << i << endl;

		if (jump_tree(generator) == 0) {  //then propose as before

			propose(p, mcmc_trees, Tc, Tstar, dict);
			//	propose_find_ratio(mcmc_trees, Tc, dict);
			n++;
		}

		else {

			jump(p, mcmc_trees, Tc, Tstar, dict, odds);

		}

		gibbs_step(Tc, dict);


		long double entr = 0;

		if (Tc.d > -1) {										//tune cutoff manually
			//entropy estimation with sample generation
			vector <short> gn = generate_from_tree(Tc, 10000, entr);
		}

		//else {

		//entropy estimation with power method
		//	entr = power_method(Tc, 10, dict);
		//}
		h.push_back(entr);
	}

	//h = 1.0 * h / (N-1);
	cout << "n is " << n << endl;
	//cout << " entropy rate is " << h << endl;
	write(h, "transfer.txt");
}

void mcmc(map< vector < vector <short>>, tree2* >& mcmc_trees, tree2 T0, map < vector <short>, node*> dict, int N) {

	// performs metropolis hastings algorithm for targeting the tree posterior
	// mcmc_trees is indexed vector of trees that occur in mcmc [max depth][TOTAL number of leaves][ith]
	// T0 is initial state of mcmc
	// dict is the dictionary based on Tmax, used to find posterior odds
	// N is number of mcmc iterations (typically  about 10^6)

	vector < vector <short>> id = treeid(T0);
	T0.f = 1;

	mcmc_trees[id] = new tree2;  //update occurrence of T0 and add it in vector
	*mcmc_trees[id] = T0;

	tree2 Tc = construct(copy(T0.t)); // current tree in mcmc T



	for (int i = 0; i < N; i++) {

		//cout << i << endl; cout << "CURRENT" << endl;
		//show_leaves(Tc.t);

		propose_find_ratio(mcmc_trees, Tc, dict);
		//cout << "PROPOSAL" << endl;
		//show_leaves(temp.t);

	}

}

void jump(double p, map< vector < vector <short>>, tree2* >& mcmc_trees, tree2& Tc, vector <tree2> Tstar, map < vector <short>, node*> dict, vector <double> odds) {

	uniform_int_distribution<int> d_j(0, k_max - 1);
	int unif = d_j(generator);
	tree2 temp = Tstar[unif];  //maybe change this to be safe (can use copy instead)
	//cout << "CURRENT" << endl;
	//show_leaves(Tc.t);


	//cout << "PROPOSAL" << endl;
	//temp.t = copy(Tstar[unif].t); // maybe se this to be asafe, but seems to make no change
	//temp = construct(temp.t);

	//show_leaves(temp.t);
	double logr = 0;
	long double podds = 0; // log-posterior odds
	long double q1 = 0;
	long double q2 = 0;
	bool indic1 = 0;

	if (mcmc_trees[treeid(Tc)] == mcmc_trees[treeid(temp)]) { // then trees are identical
		mcmc_trees[treeid(Tc)]->f++;
		///skata++;
		//cout << skata << endl;
		//cout << "SAME" << endl;
	}

	else {

		vector <short> context; int point;

		neighbours(Tc, temp, context, point);

		//cout << " point is " << point << endl; cout << "context is ";
		//for (int b = 0; b < size(context); b++) {
		//	cout << context[b];
		//}
		//cout << endl;

		if (point == -1) {// then trees are not neighbours

			int found = -1;

			for (int j = 0; j < k_max; j++) {

				if (mcmc_trees[treeid(Tc)] == mcmc_trees[treeid(Tstar[j])]) {
					found = j; j = k_max + 5;
				}
			}
			//	cout << "found is " << found << endl;

			if (found == -1) { // then T not in k-top trees

				mcmc_trees[treeid(Tc)]->f++;

			}

			else {

				logr = 1.0 * odds[found] / (1.0 * odds[unif]);

				bernoulli_distribution distr(min(1.0, logr));

				vector < vector <short>> id;

				if (distr(generator) == 0) {//then this means proposed tree is rejected
					//cout << "sample = 0" << endl;
					//Tc.f++;
					id = treeid(Tc);
					mcmc_trees[id]->f++;

					//Tc is not updated
				}

				else { // then proposal is updated

					id = treeid(temp);
					mcmc_trees[id]->f++; // it will always exist as its own of the top k

					Tc = temp;

				}



			}


		}

		else {
			//cout << skata << endl;
			//cout << "geia" << endl;

			//skata++;
			// if it turns out they are neighbours, so start cases as usually

			if (Tc.d == 0) {

				long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

				podds = prior_odds;

				for (short i = 0; i < m; i++) {
					podds = podds + dict[{i}]->le; //careful if D=0 in this case maybe error
				}
				podds = podds - dict[{}]->le;

				q1 = 1; q2 = 0.5;

			}

			else {

				if (Tc.leaves.size() + Tc.t[D].size() == 1 * pow(m, D)) {

					if (dict[temp.leaves[0]->s] != NULL) {
						podds = podds + dict[temp.leaves[0]->s]->le;
					}

					for (int ch = 0; ch < m; ch++) {

						vector <short> s = temp.leaves[0]->s;
						s.push_back(ch);
						if (dict[s] != NULL) {
							podds = podds - dict[s]->le;
						}
					}

					long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

					podds = podds + prior_odds;

					q1 = pow(1.0 * m, 1.0 * (-D + 1));
					q2 = 0.5;

				}

				else {

					if (point == 1) {

						if (dict[context] != NULL) {
							podds = podds - dict[context]->le;
						}

						for (int ch = 0; ch < m; ch++) {

							vector <short> s = context;
							s.push_back(ch);
							if (dict[s] != NULL) {
								podds = podds + dict[s]->le;
							}
						}

						long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

						podds = podds + prior_odds;

						if (temp.leaves.size() + temp.t[D].size() == 1 * pow(m, D)) { // ie if T' = full tree of depth D

							q1 = 0.5;
							q2 = pow(1.0 * m, 1.0 * (-D + 1));
							//logr = podds + 1 - 1.0 * (D - 1) * log2(m);
						}

						else {

							q1 = 0.5 / (1.0 * Tc.leaves.size());
							q2 = 0.5 / (1.0 * temp.pl.size());
							//logr = podds + log2(size(Tc.leaves)) - log2(size(temp.pl));
						}


					}

					else { // ie point = 2 and temp is shorter

						if (dict[context] != NULL) {
							podds = podds + dict[context]->le;
						}

						for (int ch = 0; ch < m; ch++) {

							vector <short> s = context;
							s.push_back(ch);
							if (dict[s] != NULL) {
								podds = podds - dict[s]->le;
							}
						}

						long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

						podds = podds + prior_odds;

						if (temp.d == 0) {
							q1 = 0.5; q2 = 1;
							//logr = podds + 1;
						}

						else {

							q1 = 0.5 / (1.0 * Tc.pl.size());
							q2 = 0.5 / (1.0 * temp.leaves.size());
							//logr = podds + log2(size(Tc.pl)) - log2(size(temp.leaves));

						}



					}


				}
			}
			for (int j = 0; j < k_max; j++) {

				if (mcmc_trees[treeid(Tc)] == mcmc_trees[treeid(Tstar[j])]) {
					indic1 = 1; j = k_max + 5;
				}
			}

			logr = pow(2.0, podds) * ((1.0 - p) * q2 + indic1 * p / (1.0 * k_max)) / ((1.0 - p) * q1 + 1.0 * p / (1.0 * k_max));

			bernoulli_distribution distr(min(1.0, logr));

			vector < vector <short>> id;

			if (distr(generator) == 0) {//then this means proposed tree is rejected
				//cout << "sample = 0" << endl;
				//Tc.f++;
				id = treeid(Tc);
				mcmc_trees[id]->f++;

				//Tc is not updated
			}

			else { // then proposal is updated

				id = treeid(temp);

				if (mcmc_trees[id] != NULL) {// ie if accepted tree temp has occured in mcmc
					mcmc_trees[id]->f++;
				}

				else { // accepted tree has't occurred before

					temp.f = 1;
					mcmc_trees[id] = new tree2;
					*mcmc_trees[id] = temp;

				}

				Tc = temp;

			}






		}
	}
}

void propose(double p, map< vector < vector <short>>, tree2* >& mcmc_trees, tree2& Tc, vector <tree2> Tstar, map < vector <short>, node*> dict) {

	tree2 temp;    // proposed tree in mcmc T'
	double logr = 0;
	long double podds = 0; // log-posterior odds
	vector <vector <short>> id;
	long double q1 = 0;
	long double q2 = 0;
	bool indic1 = 0;
	bool indic2 = 0;

	if (Tc.d == 0) { //ie if T is only the root, this is case (a)
		//cout << "geia1" << endl;
		temp.t = copy(Tc.t);
		temp.t[0][0]->leaf = 0;
		for (int ch = 0; ch < m; ch++) {

			temp.t[1].push_back(new node);
			temp.t[1][ch]->leaf = 1;
			temp.t[1][ch]->s.push_back(ch);
			temp.t[0][0]->child[ch] = temp.t[1][ch];

		}
		temp = construct(temp.t);

		long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

		podds = prior_odds;

		for (short i = 0; i < m; i++) {
			podds = podds + dict[{i}]->le; //careful if D=0 in this case maybe error
		}
		podds = podds - dict[{}]->le;

		q1 = 1; q2 = 0.5;
	}

	else {
		//cout << "geia" << endl;
		if (Tc.leaves.size() + Tc.t[D].size() == 1 * pow(m, D)) { // ie if T is the full tree of max depth D
			//cout << "geia" << endl;
			temp.t = copy(Tc.t);
			temp = construct(temp.t);

			uniform_int_distribution<int> distribution(0, temp.pl.size() - 1);
			int r = distribution(generator);
			//int r = rand() % size(temp.pl); //!!! careful if Tc.pl > rand_max = 32767, radom index for pruning


			temp.pl[r]->leaf = 1; // mark node to be a leaf

			if (dict[temp.pl[r]->s] != NULL) {
				podds = podds + dict[temp.pl[r]->s]->le;
			}

			for (int ch = 0; ch < m; ch++) {

				if (dict[temp.pl[r]->child[ch]->s] != NULL) {
					podds = podds - dict[temp.pl[r]->child[ch]->s]->le;
				}

				temp.pl[r]->child[ch] = NULL; // distruct connections with children

			}

			temp.t = copy(temp.t); // does not carry the deleted nodes as copy works with children connected to parents
			temp = construct(temp.t);

			long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

			podds = podds + prior_odds;

			q1 = pow(1.0 * m, 1.0 * (-D + 1));
			q2 = 0.5;
			//logr = podds - 1.0 + 1.0* (D - 1)* log2(m);
		}

		else {
			//cout << "geia" << endl;

			//bernoulli_distribution bindistr(0.5);
			uniform_int_distribution<int> bindistr(0, 1);
			int prune = bindistr(generator);
			//int prune = rand() % 2; // with probability 1/2 prune or add branch
			//cout << "prune is "<< prune << endl;

			if (prune == 0) { // then add a branch
				//cout << "geia" << endl;

				temp.t = copy(Tc.t);
				temp = construct(temp.t);

				uniform_int_distribution<int> distribution(0, temp.leaves.size() - 1);
				int r = distribution(generator);
				//int r = rand() % size(temp.leaves); //!!! careful if Tc.leaves > rand_max = 32767, radom index for adding branch

				temp.leaves[r]->leaf = 0;

				if (dict[temp.leaves[r]->s] != NULL) {
					podds = podds - dict[temp.leaves[r]->s]->le;
				}

				for (int ch = 0; ch < m; ch++) {

					node* new_node = new node;
					temp.t[temp.leaves[r]->s.size() + 1].push_back(new_node);
					new_node->leaf = 1;
					new_node->s = temp.leaves[r]->s;
					new_node->s.push_back(ch);
					temp.leaves[r]->child[ch] = new_node;

					if (dict[new_node->s] != NULL) {
						podds = podds + dict[new_node->s]->le;
					}

				}

				temp.t = copy(temp.t); // does not carry the deleted nodes as copy works with children connected to parents
				temp = construct(temp.t);

				long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

				podds = podds + prior_odds;

				if (temp.leaves.size() + temp.t[D].size() == 1 * pow(m, D)) { // ie if T' = full tree of depth D

					q1 = 0.5;
					q2 = pow(1.0 * m, 1.0 * (-D + 1));
					//logr = podds + 1 - 1.0 * (D - 1) * log2(m);
				}

				else {

					q1 = 0.5 / (1.0 * Tc.leaves.size());
					q2 = 0.5 / (1.0 * temp.pl.size());
					//logr = podds + log2(size(Tc.leaves)) - log2(size(temp.pl));
				}

			}

			else { // with probability 1/2 prune a branch of m nodes
				//cout << "geia" << endl;
				//show_leaves(Tc.t);
				temp.t = copy(Tc.t);
				temp = construct(temp.t);

				uniform_int_distribution<int> distribution(0, temp.pl.size() - 1);
				int r = distribution(generator);
				//int r = rand() % size(temp.pl); //!!! careful if Tc.pl > rand_max = 32767, radom index for pruning


				temp.pl[r]->leaf = 1; // mark node to be a leaf

				if (dict[temp.pl[r]->s] != NULL) {
					podds = podds + dict[temp.pl[r]->s]->le;
				}

				for (int ch = 0; ch < m; ch++) {

					if (dict[temp.pl[r]->child[ch]->s] != NULL) {
						podds = podds - dict[temp.pl[r]->child[ch]->s]->le;
					}

					temp.pl[r]->child[ch] = NULL; // distruct connections with children

				}

				temp.t = copy(temp.t); // does not carry the deleted nodes as copy works with children connected to parents
				temp = construct(temp.t);
				//	show_leaves(temp.t);

				long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

				podds = podds + prior_odds;

				if (temp.d == 0) {
					q1 = 0.5; q2 = 1;
					//logr = podds + 1;
				}

				else {

					q1 = 0.5 / (1.0 * Tc.pl.size());
					q2 = 0.5 / (1.0 * temp.leaves.size());
					//logr = podds + log2(size(Tc.pl)) - log2(size(temp.leaves));

				}



			}
		}
	}

	for (int j = 0; j < k_max; j++) {

		if (mcmc_trees[treeid(Tc)] == mcmc_trees[treeid(Tstar[j])]) {
			indic1 = 1; j = k_max + 5;
		}
	}

	for (int j = 0; j < k_max; j++) {

		if (mcmc_trees[treeid(temp)] == mcmc_trees[treeid(Tstar[j])]) {
			indic2 = 1; j = k_max + 5;
		}

	}

	logr = pow(2.0, podds) * ((1.0 - p) * q2 + indic1 * p / (1.0 * k_max)) / ((1.0 - p) * q1 + indic2 * p / (1.0 * k_max));
	//cout << logr << endl;
	bernoulli_distribution distr(min(1.0, logr));

	if (distr(generator) == 0) {//then this means proposed tree is rejected
		//cout << "sample = 0" << endl;
		//Tc.f++;
		id = treeid(Tc);
		mcmc_trees[id]->f++;

		//Tc is not updated
	}

	else { // then proposal is updated

		id = treeid(temp);

		if (mcmc_trees[id] != NULL) {// ie if accepted tree temp has occured in mcmc
			mcmc_trees[id]->f++;
		}

		else { // accepted tree has't occurred before

			temp.f = 1;
			mcmc_trees[id] = new tree2;
			*mcmc_trees[id] = temp;

		}

		Tc = temp;

	}

}

void propose_find_ratio(map< vector < vector <short>>, tree2* >& mcmc_trees, tree2& Tc, map < vector <short>, node*> dict) {
	// uses proposal to propose a new tree temp based on current tree Tc
	// and calculate metropolis hastings ratio (its logarithm)
	//T is the improper tree Tmax and dict is the dictionary, both used to find post odds
	tree2 temp;
	double logr = 0;
	long double podds = 0; // log-posterior odds
	vector <vector <short>> id;

	if (Tc.d == 0) { //ie if T is only the root, this is case (a)
		//cout << "geia1" << endl;
		temp.t = copy(Tc.t);
		temp.t[0][0]->leaf = 0;
		for (int ch = 0; ch < m; ch++) {

			temp.t[1].push_back(new node);
			temp.t[1][ch]->leaf = 1;
			temp.t[1][ch]->s.push_back(ch);
			temp.t[0][0]->child[ch] = temp.t[1][ch];

		}
		temp = construct(temp.t);

		long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

		podds = prior_odds;

		for (short i = 0; i < m; i++) {
			podds = podds + dict[{i}]->le; //careful if D=0 in this case maybe error
		}
		podds = podds - dict[{}]->le;

		logr = podds - 1.0;  // -1 is log2(1/2)
	}

	else {
		//cout << "geia" << endl;
		if (Tc.leaves.size() + Tc.t[D].size() == 1 * pow(m, D)) { // ie if T is the full tree of max depth D
			//cout << "geia" << endl;
			temp.t = copy(Tc.t);
			temp = construct(temp.t);

			uniform_int_distribution<int> distribution(0, temp.pl.size() - 1);
			int r = distribution(generator);
			//int r = rand() % size(temp.pl); //!!! careful if Tc.pl > rand_max = 32767, radom index for pruning


			temp.pl[r]->leaf = 1; // mark node to be a leaf

			if (dict[temp.pl[r]->s] != NULL) {
				podds = podds + dict[temp.pl[r]->s]->le;
			}

			for (int ch = 0; ch < m; ch++) {

				if (dict[temp.pl[r]->child[ch]->s] != NULL) {
					podds = podds - dict[temp.pl[r]->child[ch]->s]->le;
				}

				temp.pl[r]->child[ch] = NULL; // distruct connections with children

			}

			temp.t = copy(temp.t); // does not carry the deleted nodes as copy works with children connected to parents
			temp = construct(temp.t);

			long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

			podds = podds + prior_odds;

			logr = podds - 1.0 + 1.0 * (D - 1) * log2(m);
		}

		else {
			//cout << "geia" << endl;

			//bernoulli_distribution bindistr(0.5);
			uniform_int_distribution<int> bindistr(0, 1);
			int prune = bindistr(generator);
			//int prune = rand() % 2; // with probability 1/2 prune or add branch
			//cout << "prune is "<< prune << endl;

			if (prune == 0) { // then add a branch
				//cout << "geia" << endl;

				temp.t = copy(Tc.t);
				temp = construct(temp.t);

				uniform_int_distribution<int> distribution(0, temp.leaves.size() - 1);
				int r = distribution(generator);
				//int r = rand() % size(temp.leaves); //!!! careful if Tc.leaves > rand_max = 32767, radom index for adding branch

				temp.leaves[r]->leaf = 0;

				if (dict[temp.leaves[r]->s] != NULL) {
					podds = podds - dict[temp.leaves[r]->s]->le;
				}

				for (int ch = 0; ch < m; ch++) {

					node* new_node = new node;
					temp.t[temp.leaves[r]->s.size() + 1].push_back(new_node);
					new_node->leaf = 1;
					new_node->s = temp.leaves[r]->s;
					new_node->s.push_back(ch);
					temp.leaves[r]->child[ch] = new_node;

					if (dict[new_node->s] != NULL) {
						podds = podds + dict[new_node->s]->le;
					}

				}

				temp.t = copy(temp.t); // does not carry the deleted nodes as copy works with children connected to parents
				temp = construct(temp.t);

				long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

				podds = podds + prior_odds;

				if (temp.leaves.size() + temp.t[D].size() == 1 * pow(m, D)) { // ie if T' = full tree of depth D

					logr = podds + 1 - 1.0 * (D - 1) * log2(m);
				}

				else {
					//cout << size(temp.pl) << endl;
					logr = podds + log2(Tc.leaves.size()) - log2(temp.pl.size());
				}

			}

			else { // with probability 1/2 prune a branch of m nodes
				//cout << "geia" << endl;
				//show_leaves(Tc.t);
				temp.t = copy(Tc.t);
				temp = construct(temp.t);

				uniform_int_distribution<int> distribution(0, temp.pl.size() - 1);
				int r = distribution(generator);
				//int r = rand() % size(temp.pl); //!!! careful if Tc.pl > rand_max = 32767, radom index for pruning


				temp.pl[r]->leaf = 1; // mark node to be a leaf

				if (dict[temp.pl[r]->s] != NULL) {
					podds = podds + dict[temp.pl[r]->s]->le;
				}

				for (int ch = 0; ch < m; ch++) {

					if (dict[temp.pl[r]->child[ch]->s] != NULL) {
						podds = podds - dict[temp.pl[r]->child[ch]->s]->le;
					}

					temp.pl[r]->child[ch] = NULL; // distruct connections with children

				}

				temp.t = copy(temp.t); // does not carry the deleted nodes as copy works with children connected to parents
				temp = construct(temp.t);
				//	show_leaves(temp.t);

				long double prior_odds = (1.0 * temp.leaves.size() + 1.0 * temp.t[D].size() - 1.0 * Tc.leaves.size() - 1.0 * Tc.t[D].size()) * log2(alpha) + (1.0 * temp.leaves.size() - 1.0 * Tc.leaves.size()) * log2(beta);

				podds = podds + prior_odds;

				if (temp.d == 0) {
					//cout << "kl";
					logr = podds + 1;
				}

				else {

					logr = podds + log2(Tc.pl.size()) - log2(temp.leaves.size());

				}



			}
		}
	}

	logr = pow(2.0, logr); //cout << "ratio is " << r << " and min is "<< min(1.0, r) << endl;
	//cout<<min(r,r)<<endl;
	bernoulli_distribution distr(min(1.0, logr));

	if (distr(generator) == 0) {//then this means proposed tree is rejected
		//cout << "sample = 0" << endl;
		//Tc.f++;
		id = treeid(Tc);
		mcmc_trees[id]->f++;

		//Tc is not updated
	}

	else { // then proposal is updated

		id = treeid(temp);

		if (mcmc_trees[id] != NULL) {// ie if accepted tree temp has occured in mcmc
			mcmc_trees[id]->f++;
		}

		else { // accepted tree has't occurred before

			temp.f = 1;
			mcmc_trees[id] = new tree2;
			*mcmc_trees[id] = temp;

		}

		Tc = temp;

	}

}

tree copy(tree Tin) { // takes as input a tree and gives an identical copy of the tree (so that can change the 2nd one on its own)
	// also orders nodes in correct order in tree
	// works for both proper and improper trees, and trees with contexts or without contexts

	node new_node;
	vector <node*> row(1);
	row[0] = &new_node;


	tree Tout(D + 1, row);
	init_tree(Tout);

	Tout[0][0] = new node;
	*Tout[0][0] = *Tin[0][0];

	for (int i = 0; i < D; i++) {

		for (int j = 0; j < Tout[i].size(); j++) {

			for (int ch = 0; ch < m; ch++) {

				if (Tout[i][j]->child[ch] != NULL) {

					node* temp = new node;
					Tout[i + 1].push_back(temp);
					*temp = *Tout[i][j]->child[ch];
					Tout[i][j]->child[ch] = temp;



				}
			}
		}
	}

	return Tout;

}

void dictionary(map < vector <short>, node*>& dict, tree Tmax) { // construct a dictionary to look for existence of nodes in Tmax
	// using only one operation, by looking at [context] element

	dict[{}] = Tmax[0][0]; // no need to encode root node as it always exists, but can have empty context

	for (int d = 1; d < D + 1; d++) {

		for (int k = 0; k < Tmax[d].size(); k++) {

			dict[Tmax[d][k]->s] = Tmax[d][k];

		}
	}
}

vector < vector <short>> treeid(tree2 T) {

	vector < vector <short>> id;

	for (int i = 0; i < T.leaves.size(); i++) {

		id.push_back(T.leaves[i]->s);

	}

	return id;
}


vector < vector <short>> treeid2(tree T) {

	vector < vector <short>> id;

	for (int i = 0; i < D; i++) {

		for (int j = 0; j < T[i].size(); j++) {

			if (T[i][j]->leaf == 1) {
				id.push_back(T[i][j]->s);
			}

		}
	}

	return id;
}


void neighbours(tree2 T1, tree2 T2, vector <short>& context, int& point) { //returns context of node with extra branch and point to which tree has the xtra branch
	//T1 = Tc, T2 =temp
	//point = 1 means smaller neighbour is Tc
	//point = 2 means smaller neighbouts is temp
	//point = -1 means not neighbours
	//point = -2 means identical NOT USED

	tree2 t1;
	tree2 t2;

	bool found = 0;

	t1.t = copy(T1.t);
	t1 = construct(t1.t);

	for (int i = 0; i < t1.t[D].size(); i++) {

		t1.leaves.push_back(t1.t[D][i]);
	}

	t2.t = copy(T2.t);
	t2 = construct(t2.t);

	for (int i = 0; i < t2.t[D].size(); i++) {

		t2.leaves.push_back(t2.t[D][i]);
	}


	for (int i = 0; i < min(t1.leaves.size(), t2.leaves.size()); i++) {

		if (t1.leaves[i]->s.size() > t2.leaves[i]->s.size()) {

			int index = find_neighbour(t2, t1, context);

			t2.leaves[index]->leaf = 0;

			for (int ch = 0; ch < m; ch++) {

				node* new_node = new node;
				t2.t[t2.leaves[index]->s.size() + 1].push_back(new_node);
				new_node->leaf = 1;
				new_node->s = t2.leaves[index]->s;
				new_node->s.push_back(ch);
				t2.leaves[index]->child[ch] = new_node;
			}

			t2.t = copy(t2.t);
			t2 = construct(t2.t);
			for (int i = 0; i < t2.t[D].size(); i++) {

				t2.leaves.push_back(t2.t[D][i]);
			}

			found = 1;
			point = 2;
			i = min(t1.leaves.size(), t2.leaves.size()) + 5;
		}

		else {

			if (t1.leaves[i]->s.size() < t2.leaves[i]->s.size()) {

				int index = find_neighbour(t1, t2, context);

				t1.leaves[index]->leaf = 0;

				for (int ch = 0; ch < m; ch++) {

					node* new_node = new node;
					t1.t[t1.leaves[index]->s.size() + 1].push_back(new_node);
					new_node->leaf = 1;
					new_node->s = t1.leaves[index]->s;
					new_node->s.push_back(ch);
					t1.leaves[index]->child[ch] = new_node;
				}

				t1.t = copy(t1.t);
				t1 = construct(t1.t);

				for (int i = 0; i < t1.t[D].size(); i++) {

					t1.leaves.push_back(t1.t[D][i]);
				}


				found = 1;
				point = 1;
				i = min(t1.leaves.size(), t2.leaves.size()) + 5;
			}
		}
	}

	if (found == 0) {//this means trees have same number of leafs at every level, but are not identixal, hence not neighbours

		point = -1;

	}

	else {

		if (t1.leaves.size() != t2.leaves.size()) {
			point = -1;// mark not neighbours
		}

		else {

			for (int i = 0; i < t1.leaves.size(); i++) {

				if (t1.leaves[i]->s.size() != t2.leaves[i]->s.size()) {

					point = -1;
				}

				else {

					for (int p = 0; p < t1.leaves[i]->s.size(); p++) {

						if (t1.leaves[i]->s[p] != t2.leaves[i]->s[p]) {

							point = -1;
						}
					}
				}

			}
		}
	}

}

int find_neighbour(tree2 Tsmall, tree2 Tbig, vector <short>& context) {


	//returns leaves[index] of required node and sets contexts t correct value
	bool found = 0;
	int i = 0;

	while (found == 0) {

		//cout << i << endl;

		if (Tsmall.leaves[i]->s.size() != Tbig.leaves[i]->s.size()) {
			found = 1;
			context = Tsmall.leaves[i]->s;

		}

		else {

			for (int j = 0; j < Tsmall.leaves[i]->s.size(); j++) {

				if (Tsmall.leaves[i]->s[j] != Tbig.leaves[i]->s[j]) {
					//cout << "wienweifjcnwpicjnwpicjn" << endl;
					context = Tsmall.leaves[i]->s;
					found = 1;
				}
			}
		}

		//cout << found << " = found " << endl;
		i++;

	}

	return i - 1;

}

vector <short> generate_from_tree(tree2 T, int N, long double& entr) { //generate N samples from tree T and returns entropy rate estimate from sample

	vector <short> x;
	int d = T.d;
	long double h = 0;

	if (d == 0) { // if iid data no context needed

		discrete_distribution<int>  iid(T.theta[{}].begin(), T.theta[{}].end());


		for (int i = 0; i < N; i++) {
			x.push_back(iid(generator));
			h = h + log2(T.theta[{}][x[i]]);
		}

		h = -1.0 * h / N;
		//cout << "entropy of sample is " << h << endl;
		entr = h;

		return x;
	}


	// initial context drawn uniformly ar random

	uniform_int_distribution<int> init_distr(0, m - 1);

	for (int i = 0; i < d; i++) {
		x.push_back(init_distr(generator));
	}

	//generate samples 

	for (int i = d; i < N + d; i++) {

		vector <short> ct;

		for (int j = 1; j <= d; j++) {

			ct.push_back(x[i - j]);

			if (&T.theta[ct][0] != NULL) { // ie then leaf is reached

				discrete_distribution<int>  distr(T.theta[ct].begin(), T.theta[ct].end());
				x.push_back(distr(generator));
				h = h + log2(T.theta[ct][x[i]]);
				j = d + 5; // break
			}

		}

	}


	vector <short> y; // delete the initial context used to generate data, use deletions if this is faster
	for (int i = d; i < x.size(); i++) {
		y.push_back(x[i]);
	}

	h = -1.0 * h / N;
	//cout << "entropy of sample is " << h << endl;
	entr = h;

	return y;


}

vector <short> generate_from_tree2(tree T, map < vector <short>, vector <float>> theta, int N, long double& entr) { //generate N samples from tree T and returns entropy rate estimate from sample

	vector <short> x;
	int d = 0;//tree depth

	for (int i = 1; i < D + 1; i++) {
		if (T[i].size() > 0) {
			d++;
		}
	}

	long double h = 0;

	if (d == 0) { // if iid data no context needed

		discrete_distribution<int>  iid(theta[{}].begin(), theta[{}].end());


		for (int i = 0; i < N; i++) {
			x.push_back(iid(generator));
			h = h + log2(theta[{}][x[i]]);
		}

		h = -1.0 * h / N;
		//cout << "entropy of sample is " << h << endl;
		entr = h;

		return x;
	}


	// initial context drawn uniformly ar random

	uniform_int_distribution<int> init_distr(0, m - 1);

	for (int i = 0; i < d; i++) {
		x.push_back(init_distr(generator));
	}

	//generate samples 

	for (int i = d; i < N + d; i++) {

		vector <short> ct;

		for (int j = 1; j <= d; j++) {

			ct.push_back(x[i - j]);

			if (&theta[ct][0] != NULL) { // ie then leaf is reached

				discrete_distribution<int>  distr(theta[ct].begin(), theta[ct].end());
				x.push_back(distr(generator));
				h = h + log2(theta[ct][x[i]]);
				j = d + 5; // break
			}

		}

	}


	vector <short> y; // delete the initial context used to generate data, use deletions if this is faster
	for (int i = d; i < x.size(); i++) {
		y.push_back(x[i]);
	}

	h = -1.0 * h / N;
	//cout << "entropy of sample is " << h << endl;
	entr = h;

	return y;


}

void gibbs_step(tree2& T, map < vector <short>, node*> dict) { //samples thetas at leaves of tree


	// for leaves at depth <= D-1

	for (int i = 0; i < T.leaves.size(); i++) {

		vector <short> ct = T.leaves[i]->s;
		vector<double> gamma_samples;
		double sum = 0;

		if (dict[ct] != NULL) {

			for (int j = 0; j < m; j++) {

				gamma_distribution<double> distr(1.0 * dict[ct]->a[j] + 0.23, 1.0);
				gamma_samples.push_back(distr(generator));
				sum = sum + gamma_samples[j];

			}

			vector <float> theta;

			for (int j = 0; j < m; j++) {

				theta.push_back(1.0 * gamma_samples[j] / sum);
			}

			T.theta[ct] = theta;

		}

		else { // ie if leaf is not in Tmax, then all a[j]=0, sample from dirichlet prior 

			for (int j = 0; j < m; j++) {

				gamma_distribution<double> distr(0.23, 1.0);
				gamma_samples.push_back(distr(generator));
				sum = sum + gamma_samples[j];

			}

			vector <float> theta;

			for (int j = 0; j < m; j++) {

				theta.push_back(1.0 * gamma_samples[j] / sum);
			}

			T.theta[ct] = theta;

		}
	}

	//for leaves at depth D

	for (int i = 0; i < T.t[D].size(); i++) {

		vector <short> ct = T.t[D][i]->s;
		vector<double> gamma_samples;
		double sum = 0;

		if (dict[ct] != NULL) {

			for (int j = 0; j < m; j++) {

				gamma_distribution<double> distr(1.0 * dict[ct]->a[j] + 0.23, 1.0);
				gamma_samples.push_back(distr(generator));
				sum = sum + gamma_samples[j];

			}

			vector <float> theta;

			for (int j = 0; j < m; j++) {

				theta.push_back(1.0 * gamma_samples[j] / sum);
			}

			T.theta[ct] = theta;

		}

		else { // ie if leaf is not in Tmax, then all a[j]=0, sample from dirichlet prior 

			for (int j = 0; j < m; j++) {

				gamma_distribution<double> distr(0.23, 1.0);
				gamma_samples.push_back(distr(generator));
				sum = sum + gamma_samples[j];

			}

			vector <float> theta;

			for (int j = 0; j < m; j++) {

				theta.push_back(1.0 * gamma_samples[j] / sum);
			}

			T.theta[ct] = theta;

		}


	}


}

void write(vector<long double> h, string s) {

	ofstream fileOut;  // create a variable called 'fileOut'  to store info about a file.
	// It's not an int or a float, it's an ofstream; a special type 
	//  of variable used when writing to files
	fileOut.open(s); // try to open the file for writing
	if (not fileOut.good()) {
		//  if there's a problem when opening the file, print a message
		cout << "Error trying to open the file" << endl;
		return;
	}
	cout << " sizei of h is " << h.size() << endl;
	// The computer will only reach here if the file's ready to use, so use it
	for (int i = 0; i < h.size(); i++) {
		fileOut << h[i] << endl; //write to the file
	}
	fileOut.close();


}

vector< vector<float>> trans_matrix(tree2 T) {

	int d = T.d;
	int N = pow(m, d);
	vector <float> zer(N, 0);
	vector<vector<float>> mat(N, zer);

	for (int r = 0; r < N; r++) { //for each row

		//find d-ary representation of r
		vector< short> dary = dec2d(r, d);

		vector <short> ct;

		for (int l = 0; l < d; l++) {

			ct.push_back(dary[l]);

			if (&T.theta[ct][0] != NULL) { // ie then leaf is reached

				for (short j = 0; j < m; j++) {

					float theta = T.theta[ct][j];

					vector<short> temp = { j };

					int col = pow(m, d - 1) * j;

					for (int k = 0; k < d - 1; k++) {

						temp.push_back(dary[k]);
						col = col + pow(m, d - k - 2) * dary[k];
					}

					mat[r][col] = theta;

				}




				l = d + 5; // break
			}


		}

	}



	return mat;
}

vector< vector<float>> trans_matrix2(tree T, map < vector <short>, vector <float>> theta) {

	int d = 0;//tree depth
	for (int i = 1; i < D + 1; i++) {
		if (T[d].size() > 0) {
			d++;
		}
	}

	int N = pow(m, d);
	vector <float> zer(N, 0);
	vector<vector<float>> mat(N, zer);

	for (int r = 0; r < N; r++) { //for each row

		//find d-ary representation of r
		vector< short> dary = dec2d(r, d);

		vector <short> ct;

		for (int l = 0; l < d; l++) {

			ct.push_back(dary[l]);

			if (&theta[ct][0] != NULL) { // ie then leaf is reached

				for (short j = 0; j < m; j++) {

					float theta2 = theta[ct][j];

					vector<short> temp = { j };

					int col = pow(m, d - 1) * j;

					for (int k = 0; k < d - 1; k++) {

						temp.push_back(dary[k]);
						col = col + pow(m, d - k - 2) * dary[k];
					}

					mat[r][col] = theta2;

				}




				l = d + 5; // break
			}


		}

	}



	return mat;
}

vector< short> dec2d(int r, int  d) {

	vector <short> y(d, 0);
	short rem = 0; int i = d - 1;
	while (r != 0)
	{
		rem = r % m;
		r /= m;
		y[i] = rem;
		i--;
	}

	return y;

}

vector<float> vec_times_mat(vector <float> x, vector <vector <float>> P) {//computes xP, here P is transition matrix, hence square

	int N = x.size();
	vector <float> y;

	for (int i = 0; i < N; i++) {

		long double sum = 0;

		for (int j = 0; j < N; j++) {

			sum = sum + P[j][i] * x[j];

		}

		y.push_back(sum);
	}

	return y;
}

long double power_method(tree2 T, int n, map < vector <short>, node*> dict) { // power_method to calculate stationaty distribution and entropy rate, n is number of iterations 

	double h = 0;
	int d = T.d;



	//cout << "geua  " << d<< endl;

	if (d == 0) {

		for (int i = 0; i < T.theta[{}].size(); i++) {
			long double p = T.theta[{}][i];
			if (p > 0) {
				h = h + p * log2(p);
			}
		}
		//cout << "iid" << endl;

		return -h;

	}

	vector<vector<float>> mat = trans_matrix(T);

	int N = mat.size();
	//cout << N << endl;

	vector <float> init(N, 0); // use initial guess, either uniform or counts as from Tmax

	long double sum = 0;

	for (int i = 0; i < N; i++) {

		vector<short> mary = dec2d(i, d);

		if (dict[mary] != NULL) {

			for (int j = 0; j < m; j++) {
				init[i] = init[i] + dict[mary]->a[j] + 0.5; // sum of as + 1/2 because thetas from posterrior with pseudocounts
			}

			sum = sum + init[i];

		}

		else {
			init[i] = 0.5 * m;
			sum = sum + init[i];

		}
	}

	for (int i = 0; i < N; i++) {

		init[i] = init[i] / sum;

	}

	//vector<float> init2(N, 1.0* 1/N);//uncomment these 2 to use uniform priors
	//init = init2;



	//for (int i = 0; i < N; i++) { cout << init[i]<<"  "; } cout << endl;

	//power method computed here

	for (int i = 0; i < n; i++) {

		init = vec_times_mat(init, mat);

	}

	long double sum2 = 0;

	for (int i = 0; i < N; i++) {

		sum2 = sum2 + init[i];

	}

	for (int i = 0; i < N; i++) {

		init[i] = init[i] / sum2;

	}

	//for (int i = 0; i < N; i++) { cout << init[i]<<"  "; } cout << endl;
	//cout<< "stationaty vector is " <<init<<endl;

	//long double margh = -1.0 * init[0] * log2(init[0]) - 1.0 * init[1] * log2(init[1]);
	//cout << margh << endl;

	double p1 = 0;
	double p2 = 0;

	for (int b = 0; b < N / 2; b++) {
		p1 = p1 + init[b];
	}
	for (int b = N / 2; b < N; b++) {
		p2 = p2 + init[b];
	}

	long double margh = 0;
	if (p1 > 0) {
		margh = margh - 1.0 * p1 * log2(p1);
	}
	if (p2 > 0) {
		margh = margh - 1.0 * p2 * log2(p2);
	}

	//cout << margh << endl;

	return margh; // this is for marginal entropy

	//now calculate entropy rate

	for (int i = 0; i < N; i++) {

		long double pij = 0;

		for (int j = 0; j < N; j++) {

			if (mat[i][j] > 0) {
				pij = pij + mat[i][j] * log2(mat[i][j]);
			}
		}

		h = h + init[i] * pij;


	}

	//cout << -h << endl;


	return -h; //this is for entropy rate



}

long double power_method2(tree T, map < vector <short>, vector <float>> theta, int n, map < vector <short>, node*> dict) { // power_method to calculate stationaty distribution and entropy rate, n is number of iterations 

	long double h = 0;
	int d = 0;

	//cout << "geua  " << d<< endl;

	for (int i = 1; i < D + 1; i++) {
		if (T[i].size() > 0) {
			d++;
		}
	}

	if (T[0][0]->leaf == 1) {

		for (int i = 0; i < theta[{}].size(); i++) {
			long double p = theta[{}][i];
			if (p > 0) {
				h = h + p * log2(p);
			}
		}

		return -h;

	}

	vector<vector<float>> mat = trans_matrix2(T, theta);

	int N = mat.size();

	vector <float> init(N, 0); // use initial guess, either uniform or counts as from Tmax

	long double sum = 0;

	for (int i = 0; i < N; i++) {

		vector<short> mary = dec2d(i, d);

		if (dict[mary] != NULL) {

			for (int j = 0; j < m; j++) {
				init[i] = init[i] + dict[mary]->a[j] + 0.5; // sum of as + 1/2 because thetas from posterrior with pseudocounts
			}

			sum = sum + init[i];

		}

		else {
			init[i] = 0.5 * m;
			sum = sum + init[i];

		}
	}

	for (int i = 0; i < N; i++) {

		init[i] = init[i] / sum;

	}

	//vector<float> init2(N, 1.0* 1/N);//uncomment these 2 to use uniform priors
	//init = init2;

	//for (int i = 0; i < N; i++) { cout << init[i]<<"  "; } cout << endl;

	//power method computed here

	for (int i = 0; i < n; i++) {

		init = vec_times_mat(init, mat);

	}

	long double sum2 = 0;

	for (int i = 0; i < N; i++) {

		sum2 = sum2 + init[i];

	}

	for (int i = 0; i < N; i++) {

		init[i] = init[i] / sum2;

	}

	//for (int i = 0; i < N; i++) { cout << init[i]<<"  "; } cout << endl;
	//cout<< "stationaty vector is " <<init<<endl;

	//now calculate entropy rate

	for (int i = 0; i < N; i++) {

		long double pij = 0;

		for (int j = 0; j < N; j++) {

			if (mat[i][j] > 0) {
				pij = pij + mat[i][j] * log2(mat[i][j]);
			}
		}

		h = h + init[i] * pij;


	}

	//cout << -h << endl;
	return -h;



}

tree sample(map < vector <short>, node*> dict) { //gives 1 sample from tree posterior
	//cout << "sample" << endl;
	tree T = {};
	vector <node*> init = {};

	for (int d = 0; d < D + 1; d++) {
		T.push_back(init);
	}

	T[0].push_back(new node);


	for (int d = 0; d < D; d++) {

		for (int k = 0; k < T[d].size(); k++) {

			long double lp;

			if (dict[T[d][k]->s] != NULL) {//if node in Tmax

				lp = log2(beta) + dict[T[d][k]->s]->le - dict[T[d][k]->s]->lw; // with this probablity mark node to be leaf

			}

			else { lp = log2(beta); }


			bernoulli_distribution distr(pow(2.0, lp));

			if (distr(generator) == 1) { //then node is marked as leaf

				T[d][k]->leaf = 1;

			}

			else { // add all its children to the tree

				for (int j = 0; j < m; j++) {

					node* new_node = new node;
					new_node->s = T[d][k]->s;
					new_node->s.push_back(j);
					T[d][k]->child[j] = new_node;
					T[d + 1].push_back(new_node);

				}


			}



		}





	}

	for (int k = 0; k < T[D].size(); k++) {

		T[D][k]->leaf = 1;

	}

	//cout << "tree sampled" << endl;
	//show_tree(T);
	return T;
}

vector <long double> entr(int N, map < vector <short>, node*> dict) { // could optimise by sampling directly the thetas when 
	// node is a leaf: then no need to scan tree again
	vector <long double> h;


	for (int i = 0; i < N; i++) {

		tree T = sample(dict);
		map < vector <short>, vector <float>> theta = gibbs_step2(T, dict);

		long double entr = 0;

		vector <short> gn = generate_from_tree2(T, theta, 10000, entr);

		//cout << entr << endl;
		//else {

		//entropy estimation with power method
		//	entr = power_method(Tc, 10, dict);
		//}
		h.push_back(entr);
	}


	write(h, "transfer.txt");





	return h;

}

vector <long double> entr2(int N, map < vector <short>, node*> dict) { // could optimise by sampling directly the thetas when 
	// node is a leaf: then no need to scan tree again
	vector <long double> h;


	for (int i = 0; i < N; i++) {

		tree T = sample(dict);
		//cout << "prin" << endl;
		tree2 Tc = construct(T);
		gibbs_step(Tc, dict);
		long double entr = 0;

		vector <short> gn = generate_from_tree(Tc, 10000, entr);
		//cout << "meta" << endl;

		//else {

		//entropy estimation with power method
		//	entr = power_method(Tc, 10, dict);
		//}
		h.push_back(entr);
	}


	write(h, "transfer.txt");





	return h;

}

map < vector <short>, vector <float>> gibbs_step2(tree T, map < vector <short>, node*> dict) {


	map < vector <short>, vector <float>> theta;



	for (int i = 0; i < D + 1; i++) {

		for (int j = 0; j < T[i].size(); j++) {

			if (T[i][j]->leaf == 1) {

				vector <short> ct = T[i][j]->s;
				vector<double> gamma_samples;
				double sum = 0;

				if (dict[ct] != NULL) {

					for (int j = 0; j < m; j++) {

						gamma_distribution<double> distr(dict[ct]->a[j] + 0.5, 1.0);
						gamma_samples.push_back(distr(generator));
						sum = sum + gamma_samples[j];

					}

					vector <float> theta2;

					for (int j = 0; j < m; j++) {

						theta2.push_back(1.0 * gamma_samples[j] / sum);
					}

					theta[ct] = theta2;

				}

				else { // ie if leaf is not in Tmax, then all a[j]=0, sample from dirichlet prior 

					for (int j = 0; j < m; j++) {

						gamma_distribution<double> distr(0.5, 1.0);
						gamma_samples.push_back(distr(generator));
						sum = sum + gamma_samples[j];

					}

					vector <float> theta2;

					for (int j = 0; j < m; j++) {

						theta2.push_back(1.0 * gamma_samples[j] / sum);
					}

					theta[ct] = theta2;

				}
			}
		}
	}





	return theta;
}


void sample_trees(int N, map < vector <short>, node*> dict, map< vector < vector <short>>, tree2* >& mcmc_trees) {


	//ofstream myfile;	myfile.open("transfer_depth.txt"); 
	//ofstream myfile2;   myfile2.open("transfer_leaves.txt");

	for (int i = 0; i < N; i++) {

		tree T = sample(dict);
		vector <vector <short>> id = treeid2(T);

		//tree2 Tc = construct(T);   //use these two insteand of line above when want samples from depth, leaves
		//vector <vector <short>> id = treeid(Tc);

		if (mcmc_trees[id] != NULL) {// ie if accepted tree temp has occured in mcmc
			mcmc_trees[id]->f++;
		}

		else { // accepted tree has't occurred before

			tree2 Tc = construct(T); // comment this out when samples needed fro depth, leaves
			Tc.f = 1;
			mcmc_trees[id] = new tree2;
			*mcmc_trees[id] = Tc;

		}


		//myfile << Tc.d << endl;
		//myfile2 << Tc.leaves.size() + Tc.t[D].size() << endl;

	}

	//myfile.close();
	//myfile2.close();


}


void marg_entr(int N, map < vector <short>, node*> dict) { // could optimise by sampling directly the thetas when 
	// node is a leaf: then no need to scan tree again


	ofstream myfile;	myfile.open("transfer_margh.txt");


	for (int i = 0; i < N; i++) {

		tree T = sample(dict);
		//cout << "prin" << endl;
		tree2 Tc = construct(T);
		gibbs_step(Tc, dict);
		long double entr = 0;

		long double margh = power_method(Tc, 10, dict);
		//cout << "meta" << endl;
		myfile << margh << endl;

		//else {

		//entropy estimation with power method
		//	entr = power_method(Tc, 10, dict);
		//}

	}

	myfile.close();


}

void p_est(node* N) { // this calculates the estimated probability N->le from the sums

	long double lp;


	//int Bs = N->Bs;
	//double s1 = N->s1; //these are not needed
	Matrix<double, ar_p + 1, 1> s2; //store matrices for eigen-library
	Matrix<double, ar_p + 1, ar_p + 1> s3;//this is symmetric, positive definite (or semi-definite, but probably iinvertible)
	Matrix<double, ar_p + 1, ar_p + 1> s3_inv;//inverse matrix
	Matrix<double, ar_p + 1, ar_p + 1> identit;//identity matrix

	identit = identit.Identity();

	for (int i = 0; i < ar_p + 1; i++) {
		s2(i, 0) = N->s2[i];
	}

	for (int i = 0; i < ar_p + 1; i++) {
		for (int j = 0; j < ar_p + 1; j++) {
			s3(i, j) = N->s3[i][j];
		}
	}


	long double A_B = N->s1 + mu_0.transpose() * sigma_0.inverse() * mu_0;
	Matrix<double, ar_p + 1, 1> t = s2 + sigma_0.inverse() * mu_0;
	Matrix<double, ar_p + 1, ar_p + 1> S2 = s3 + sigma_0.inverse();




	A_B = A_B - t.transpose() * S2.inverse() * t; // A_B is A+B

	lp = tau * log2(lambda) - (tau + 0.5 * N->Bs) * log2(lambda + 0.5 * A_B);




	Matrix<double, ar_p + 1, ar_p + 1> I_matrix;
	I_matrix = I_matrix.Identity();

	Matrix<double, ar_p + 1, ar_p + 1> J = s3 * sigma_0 + I_matrix;
	lp = lp - 0.5 * log2(J.determinant()) - 0.5 * N->Bs * log2(2.0 * M_PI);


	lp = lp + 1 / log(2.0) * (lgamma(tau + 0.5 * N->Bs) - lgamma(tau));

	N->le = lp;

}


void post_param(tree T) {

	for (int d = 0; d < D + 1; d++) {

		for (int k = 0; k < T[d].size(); k++) {

			if (T[d][k]->leaf == 1) {

				node* N = T[d][k];

				Matrix<double, ar_p + 1, 1> s2; //store matrices for eigen-library
				Matrix<double, ar_p + 1, ar_p + 1> s3;//this is symmetric, positive definite (or semi-definite, but probably iinvertible)
				Matrix<double, ar_p + 1, ar_p + 1> s3_inv;//inverse matrix
				Matrix<double, ar_p + 1, ar_p + 1> identit;//identity matrix

				identit = identit.Identity();

				for (int i = 0; i < ar_p + 1; i++) {
					s2(i, 0) = N->s2[i];
				}

				for (int i = 0; i < ar_p + 1; i++) {
					for (int j = 0; j < ar_p + 1; j++) {
						s3(i, j) = N->s3[i][j];
					}
				}

				//posterior now is a t distribution
				Matrix<double, ar_p + 1, ar_p + 1> Sigma_inv = s3 + sigma_0.inverse(); //not in equations exactly like this






				Matrix<double, ar_p + 1, 1> m = Sigma_inv.inverse() * (sigma_0.inverse() * mu_0 + s2); // mean

				long double A_B = N->s1 + mu_0.transpose() * sigma_0.inverse() * mu_0;
				Matrix<double, ar_p + 1, 1> t = s2 + sigma_0.inverse() * mu_0;
				//Matrix<double, ar_p+1, ar_p+1> S2 = s3 + sigma_0.inverse();

				A_B = A_B - t.transpose() * m; // A_B is A+B


				double nu = 2 * tau + 1.0 * N->Bs;

				//Matrix<double, ar_p, ar_p> P_inv = Sigma_inv * (nu / (2.0* lambda + A_B)); // this is inverse scale matrix, uncomment to calculateit and print entries


				//print mean
				cout << "posterior mean " << endl;
				for (int j = 0; j < ar_p + 1; j++) {
					cout << m(j) << "     ";
				}
				cout << endl;


				//cout << "degrees of freedom v = " << nu << endl;


				//posterior for sigma ^2 is inv-gamma with parameters 0.5* nu, lambda + 0.5 * A + 0.5 * B

				cout << "posterior mode of sigma^2 is " << (2.0 * lambda + A_B) / (2.0 * tau + 1.0 * N->Bs + 2.0) << endl;


				cout << "Bs/2 is " << 0.5 * N->Bs << endl;
				cout << "Ds/2 is " << 0.5 * A_B << endl;

				for (int w = 0; w < N->s.size(); w++) {
					cout << N->s[w];
				}
				cout << endl;




			}

		}

	}




}











void log_loss(vector <double> xn, int train_size) { //this calculates log-loss exactly as discrete case, using full predictive distribution (not possible in reality to achieve this)

	//This function computes the (cumulatve) log-loss at each time-step in the test set.
	//The output is written in the file "log_loss.txt"
	//The input xn here is the whole dataset and train_size is the size of the training set.

	vector <long double> log_loss;   // Store the the log-loss at each time-step


	//1. First perform CTW for the training sequence

	node new_node;
	vector <node*> row(1);
	row[0] = &new_node;


	tree T(D + 1, row);
	init_tree(T);
	tree Test = T;

	for (int i = D; i < train_size; i++) {

		double s = xn[i];          // current symbol

		vector<double> x_tilde(ar_p + 1);     //continuous context needed for sums
		x_tilde[0] = 1.0;
		for (int j = 1; j < ar_p + 1; j++) {
			x_tilde[j] = xn[i - j];
		}

		vector <short> ct(D);     // current context



		for (int j = 0; j < D; j++) { //here use QUANTISED version for disrete context

			//ct[j] = xn[i - j - 1];  // sets context
			//cout << ct[j];   // prints context
			//if (xn[i - j] > xn[i - j - 1]) {
			//	ct[j] = 1;
			//}
			//else { ct[j] = 0; }

			if (xn[i - j - 1] > 0) {
				ct[j] = 1;
			}
			else { ct[j] = 0; }
		}
		//cout << endl;

		update(T, s, ct, x_tilde);

	}

	cout << "Tree was built" << endl;
	//show_tree(T);

	// HERE EXTRA FOR CONTINUOUS: from sums calculate N->le for all nodes ot Tmax (still improper)

	for (int d = 0; d < D + 1; d++) {
		for (int k = 0; k < T[d].size(); k++) {
			p_est(T[d][k]);
		}
	}

	long double init_ctw = rma3(T);     // Store the prior-predictive likelihood in the training set


	//2.Then evaluate the log-loss incurred by prediction in the test set by perfoming CTW sequentially


	for (int i = train_size; i < xn.size(); i++) {
		double s = xn[i];          // current symbol

		vector<double> x_tilde(ar_p + 1);     //continuous context needed for sums
		x_tilde[0] = 1.0;
		for (int j = 1; j < ar_p + 1; j++) {
			x_tilde[j] = xn[i - j];
		}

		vector <short> ct(D);     // current context



		for (int j = 0; j < D; j++) { //here use QUANTISED version for disrete context

			//ct[j] = xn[i - j - 1];  // sets context
			//cout << ct[j];   // prints context
			//if (xn[i - j] > xn[i - j - 1]) {
			//	ct[j] = 1;
			//}
			//else { ct[j] = 0; }

			if (xn[i - j - 1] > 0) {
				ct[j] = 1;
			}
			else { ct[j] = 0; }
		}
		//cout << endl;

		update(T, s, ct, x_tilde);

		// Here to perform CTW we only have to look at the D+1 contexts


		vector <node*> nodes_ct; //pointers to these nodes in Tmax (nodes already exist)

		nodes_ct.push_back(T[0][0]);

		for (int j = 1; j < D + 1; j++) {

			nodes_ct.push_back(nodes_ct[j - 1]->child[ct[j - 1]]); // this sets node pointers

		}

		for (int d = D; d > -1; d--) {
			p_est(nodes_ct[d]); //update p_est for updated contexts
		}







		for (int d = D; d > -1; d--) {           // loop over levels, same as before


			if (d == D) {                   // if at max depth, node is a leaf
				nodes_ct[d]->lw = nodes_ct[d]->le;
			}

			else {                         // if node is not a leaf

				long double sum = 0;

				for (int ch = 0; ch < m; ch++) {

					if (nodes_ct[d]->child[ch] != NULL) {           // if child #ch exists
						sum = sum + nodes_ct[d]->child[ch]->lw;     // calculate sum of Le(s)

					}

				}

				//calculate weighted log-prob in two cases (for numerical precision)

				long double delta = nodes_ct[d]->le - sum + log2(beta) - log2(1.0 - beta);
				if (delta < 30) {

					nodes_ct[d]->lw = log2(1.0 - beta) + sum + log2(1.0 + pow(2.0, delta));

				}
				else {
					nodes_ct[d]->lw = log2(beta) + nodes_ct[d]->le + log2(exp(1)) * (pow(2.0, -delta) - pow(2.0, -2.0 * delta - 1));

				}

			}

		}

		//END of sequential CTW


		log_loss.push_back(init_ctw - T[0][0]->lw); // this evaluates: -log P(x_{n+i}| x_1^n) = P(x_1^n)/P(x_1^{n+i)
		// i.e. the cumulative log-loss up to symbol i of the test set

	}


	for (int i = 0; i < log_loss.size(); i++) { log_loss[i] = log_loss[i] * log(2.0); } // convert log2 to ln

	//3. Finally, write the log-loss for each timestep in the file "log_loss.txt"

	ofstream fileOut;
	fileOut.open("log_loss.txt");
	for (int i = 0; i < log_loss.size(); i++) {
		fileOut << log_loss[i] << endl;
	}
	fileOut.close();

	cout << "Log-loss was written in file 'log_loss.txt'" << endl;

}

long double predict_one_step(tree T, vector <short> ct, vector<double> x_tilde) { //x is real value used for log-loss, function used in predict_online using T_map

	node* N = T[0][0];

	for (int j = 0; j < D; j++) {

		if (N->leaf == 1) {
			j = D + 5;//break
		}

		else {
			N = N->child[ct[j]];
		}

	}


	Matrix<double, ar_p + 1, 1> s2; //store matrices for eigen-library
	Matrix<double, ar_p + 1, ar_p + 1> s3;//this is symmetric, positive definite (or semi-definite, but probably iinvertible)
	Matrix<double, ar_p + 1, ar_p + 1> s3_inv;//inverse matrix
	Matrix<double, ar_p + 1, ar_p + 1> identit;//identity matrix

	identit = identit.Identity();

	for (int i = 0; i < ar_p + 1; i++) {
		s2(i, 0) = N->s2[i];
	}

	for (int i = 0; i < ar_p + 1; i++) {
		for (int j = 0; j < ar_p + 1; j++) {
			s3(i, j) = N->s3[i][j];
		}
	}

	//posterior now is a t distribution
	Matrix<double, ar_p + 1, ar_p + 1> Sigma_inv = s3 + sigma_0.inverse(); //not in equations exactly like this






	Matrix<double, ar_p + 1, 1> m = Sigma_inv.inverse() * (sigma_0.inverse() * mu_0 + s2); // mean

	long double A_B = N->s1 + mu_0.transpose() * sigma_0.inverse() * mu_0;
	Matrix<double, ar_p + 1, 1> t = s2 + sigma_0.inverse() * mu_0;
	//Matrix<double, ar_p, ar_p> S2 = s3 + sigma_0.inverse(); // this is Sigma_inv

	A_B = A_B - t.transpose() * Sigma_inv.inverse() * t; // A_B is A+B


	double nu = 2 * tau + 1.0 * N->Bs;

	long double y = 0.0;
	for (int k = 0; k < ar_p + 1; k++) {
		//if (m(k)> 0.05 || m(k)<-0.05) {
		y = y + m(k) * x_tilde[k];
		//}
	}

	//cout << m << endl;



	//cout << y << endl;
	return y;

}



void pred_mse(vector <double> xn, int train_size) { //this calculates log-loss exactly as discrete case, using full predictive distribution (not possible in reality to achieve this)

	node new_node;
	vector <node*> row(1);
	row[0] = &new_node;


	tree T(D + 1, row);
	init_tree(T);
	tree Test = T;
	//cout << endl<< endl<< size(xn);

	// initialise array to keep k-mapt trees

	vector <tree> trees(k_max, T);
	vector <tree> trees_0 = trees;

	// find preprocessing nodes at each depth

	vector <node*> init; // pointers for nodes of pre-processing stage (not needed  for root node)

	if (D > 0) {

		for (short d = 0; d < D; d++) {
			init.push_back(new node); // initialise them
		}
		preproc(init);

	}

	// update for each sequence symbol
	for (int i = D; i < train_size; i++) { // or D+1 here
		//cout << xn[i] << endl; //prints sequence

		double s = xn[i];          // current symbol
		vector<double> x_tilde(ar_p + 1);     //continuous context needed for sums
		x_tilde[0] = 1.0;
		for (int j = 1; j < ar_p + 1; j++) {
			x_tilde[j] = xn[i - j];
		}

		vector <short> ct(D);     // current context

		//cout << endl << "symbol " << i << ", with context ";

		for (int j = 0; j < D; j++) { //here use QUANTISED version for disrete context




			//cout << xn[-1] << "     ";
			if (xn[i - j - 1] > 0.15) {
				ct[j] = 1;
			}
			else { ct[j] = 0; }
		}





		//cout << endl;

		update(T, s, ct, x_tilde);

	}
	cout << xn.size() << endl;
	cout << "Tree was built" << endl;
	//show_tree(T);

	// HERE EXTRA FOR CONTINUOUS: from sums calculate N->le for all nodes ot Tmax (still improper)

	for (int d = 0; d < D + 1; d++) {
		for (int k = 0; k < T[d].size(); k++) {
			p_est(T[d][k]);
		}
	}


	//mle2(T);

	//show_tree(T);
	//rma3(T);

	vector<double> odds(k_max, 0.0);
	kmapt(T, trees, init, odds);

	long double mse = 0.0;
	long double mse2 = 0.0;

	//post_param(trees[0]);

	//2.Then evaluate the log-loss incurred by prediction in the test set by updating CTW sequentially


	for (int i = train_size; i < xn.size(); i++) {

		//cout << i << endl;

		double s = xn[i];          // current symbol

		vector<double> x_tilde(ar_p + 1);     //continuous context needed for sums
		x_tilde[0] = 1.0;
		for (int j = 1; j < ar_p + 1; j++) {
			x_tilde[j] = xn[i - j];
		}

		vector <short> ct(D);     // current context



		for (int j = 0; j < D; j++) { //here use QUANTISED version for disrete context




			if (xn[i - j - 1] > 0.15) {
				ct[j] = 1;
			}
			else { ct[j] = 0; }
		}


		long double y0 = predict_one_step(trees[0], ct, x_tilde); ////USE THISSSSS FOR ONE STEP
		//long double y1 = predict_one_step(trees[1], ct, x_tilde);
		//long double y2 = predict_one_step(trees[2], ct, x_tilde);


		mse = mse + (y0 - xn[i + 0]) * (y0 - xn[i + 0]);

		//cout << xn[i + 3] << endl;

		update(T, s, ct, x_tilde);

		// Here to perform CTW we only have to look at the D+1 contexts


		vector <node*> nodes_ct; //pointers to these nodes in Tmax (nodes already exist)

		nodes_ct.push_back(T[0][0]);

		for (int j = 1; j < D + 1; j++) {

			nodes_ct.push_back(nodes_ct[j - 1]->child[ct[j - 1]]); // this sets node pointers

		}

		for (int d = D; d > -1; d--) {
			p_est(nodes_ct[d]); //update p_est for updated contexts
		}



		for (int d = D; d > -1; d--) {           // loop over levels, same as before


			if (d == D) {                   // if at max depth, node is a leaf
				nodes_ct[d]->lw = nodes_ct[d]->le;
			}

			else {                         // if node is not a leaf

				long double sum = 0;

				for (int ch = 0; ch < m; ch++) {

					if (nodes_ct[d]->child[ch] != NULL) {           // if child #ch exists
						sum = sum + nodes_ct[d]->child[ch]->lw;     // calculate sum of Le(s)

					}

				}

				//calculate weighted log-prob in two cases (for numerical precision)

				long double delta = nodes_ct[d]->le - sum + log2(beta) - log2(1.0 - beta);
				if (delta < 30) {

					nodes_ct[d]->lw = log2(1.0 - beta) + sum + log2(1.0 + pow(2.0, delta));

				}
				else {
					nodes_ct[d]->lw = log2(beta) + nodes_ct[d]->le + log2(exp(1)) * (pow(2.0, -delta) - pow(2.0, -2.0 * delta - 1));

				}

			}

		}

		//END of sequential CTW


		//3. Find top-k trees again
		trees = trees_0; //temporary trees


		//kmapt(T, trees, init, odds); // kmpt without rma here

		vector <double> lm_init = { 0 };
		matrix c_init;

		for (int d = 0; d < D + 1; d++) {
			for (int k = 0; k < T[d].size(); k++) {
				T[d][k]->lm = lm_init;
				T[d][k]->c = c_init;
			}
		}


		kmapt_forw(T, init);

		kmapt_back(init, T, trees);

		for (int i = 0; i < k_max; i++) {

			label(trees[i]);
		}



	}

	cout << endl;
	cout << "mse is " << mse / (xn.size() - train_size) << endl;



}


long double predict_multi_step(tree T, vector <short> ct, vector<double> x_tilde, short step, int N_samples) {

	long double sum = 0.0;


	for (int n_traj = 0; n_traj < N_samples; n_traj++) {

		vector <short> ctc = ct;
		vector<double> x_tildec = x_tilde;
		double sample = 0.0;

		for (int n_step = 0; n_step < step; n_step++) {

			node* N = T[0][0];

			for (int j = 0; j < D; j++) {

				if (N->leaf == 1) {
					j = D + 5;//break
				}

				else {
					N = N->child[ctc[j]];
				}

			}


			Matrix<double, ar_p + 1, 1> s2; //store matrices for eigen-library
			Matrix<double, ar_p + 1, ar_p + 1> s3;//this is symmetric, positive definite (or semi-definite, but probably iinvertible)
			Matrix<double, ar_p + 1, ar_p + 1> s3_inv;//inverse matrix
			Matrix<double, ar_p + 1, ar_p + 1> identit;//identity matrix

			identit = identit.Identity();

			for (int i = 0; i < ar_p + 1; i++) {
				s2(i, 0) = N->s2[i];
			}

			for (int i = 0; i < ar_p + 1; i++) {
				for (int j = 0; j < ar_p + 1; j++) {
					s3(i, j) = N->s3[i][j];
				}
			}

			//posterior now is a t distribution
			Matrix<double, ar_p + 1, ar_p + 1> Sigma_inv = s3 + sigma_0.inverse(); //not in equations exactly like this


			Matrix<double, ar_p + 1, 1> m = Sigma_inv.inverse() * (sigma_0.inverse() * mu_0 + s2); // mean

			long double A_B = N->s1 + mu_0.transpose() * sigma_0.inverse() * mu_0;
			Matrix<double, ar_p + 1, 1> t = s2 + sigma_0.inverse() * mu_0;
			//Matrix<double, ar_p, ar_p> S2 = s3 + sigma_0.inverse(); // this is Sigma_inv

			A_B = A_B - t.transpose() * Sigma_inv.inverse() * t; // A_B is A+B


			double nu = 2 * tau + 1.0 * N->Bs;
			long double s_hat = (2.0 * lambda + A_B) / (2.0 * tau + 1.0 * N->Bs + 2.0);


			long double m_current = 0.0;

			for (int k = 0; k < ar_p + 1; k++) {
				//if (m(k)> 0.05 || m(k)<-0.05) {
				m_current = m_current + m(k) * x_tildec[k];
				//}
			}

			normal_distribution<double> n_distr(m_current, s_hat);
			sample = n_distr(generator);

			vector<short> ct_temp = ctc;

			for (int j = 0; j < ct.size() - 1; j++) {
				ct_temp[j + 1] = ctc[j];
			}

			if (sample > 0.2) {
				ct_temp[0] = 1;
			}
			else { ct_temp[0] = 0; }

			ctc = ct_temp;

			vector <double> x_tilde_temp = x_tildec;

			for (int j = 1; j < x_tilde.size() - 1; j++) {
				x_tilde_temp[j + 1] = x_tildec[j];
			}

			x_tilde_temp[1] = sample;

			x_tildec = x_tilde_temp;





		}

		//cout<<"sample is " << sample<<endl; // here sample is the sample of xT+l
		sum = sum + sample;

	}

	//cout << "geia" << endl;
	return sum / (1.0 * N_samples);

}
