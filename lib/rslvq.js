/*
*   Adaptive Robust Soft Learning Vector Quantization
*
*   References
*   ----------
*    .. [1] Heusinger, M., Raab, C., Schleif, F.M.: Passive concept drift
*    handling via momentum based robust soft learning vector quantization.\
*    In: Vellido, A., Gibert, K., Angulo, C., Martı́n Guerrero, J.D. (eds.)
*    Advances in Self-Organizing Maps, Learning Vector Quantization,
*    Clustering and Data Visualization. pp. 200–209. Springer International
*    Publishing, Cham (2020)
*    .. [2] Sambu Seo and Klaus Obermayer. 2003. Soft learning vector
*    quantization. Neural Comput. 15, 7 (July 2003), 1589-1604
 */

const nj = require(`numjs`);

class RSLVQ {

    constructor(options) {
        this.options = options || {};
        this.prototypesPerClass = options.prototypesPerClass || 2;
        this.sigma = options.sigma || 1.0;
        this.epsilon = 0.00000001;
        this.initialPrototypes = options.initialPrototypes || false;
        this.gamma = options.gamma || 0.9;
        this.initialFit = true;
    } // endConstructor

    async partialFit(X, y) {
        if (this.initialFit) {
            // its the first fit we have to do init prototypes
            this.validatePrototypes(X, y);
        } // endIf

        await self.optimize();
    } // endPartialFit

    async validatePrototypes(X, y) {

    } // endValidatePrototypes
} // endClassRSLVQ

module.exports = RSLVQ;