# Differences Between Optickle and Finesse

Optickle and Finesse are fairly similar on the surface, and PyTickle aims to reduce the differences as much as possible. For both simulation packages components are added and connected with links to create optomechanical models. Nevertheless, there are significant differences in the details of the implementation. Here is an incomplete list. This list is meant as a reference; see the examples for more context and explanations.

## Differences in model definitions

 * One of the main differences between Optickle and Finesse is the way nodes are connected. Finesse uses spaces which allow fields to propagate in both directions while Optickle uses links which only allow fields to propagate in one direction. So while the Finesse code
 ```python
fin.addSpace(kat, 'IX_fr', 'EX_fr', Lcav)
```
makes a cavity between the mirrors `IX` and `EX`, the Optickle code
```python
opt.addLink('IX', 'fr', 'EX', 'fr', Lcav)
opt.addLink('EX', 'fr', 'IX', 'fr', Lcav)
```
makes the same cavity. Faraday isolators can effectively be created in Optickle by appropriate use of the one-way links while Finesse has separate Faraday isolator components that can be added with `addFaradayIsolator`.
 * Components in Optickle come with their own node names while Finesse allows you to name the nodes whatever you want. PyTickle's Finesse model building functions enforce standardized node naming conventions, similar to Optickle, which the user does not have to think about. In addition to making the nodes standard, descriptive, and easier to remember, this has the advantage of making the model behavior independent of the order in which components are defined or in which spaces are added. In particular
   * The radius of curvature of a mirror is always the physical radius of curvature. With classic Finesse code, the sign of the ROCs depends on the order in which the nodes are defined. As a consequence, the sign of the ROC of the input coupler of optical cavities is often opposite to that of the real mirror. This will never happen if you only use PyTickle's model building functions.
   * The sign of the tunings of optics is also always the same when added with PyTickle functions but again depends on the order in which nodes are defined with classic Finesse code.
 * Microscopic tunings are in meters in Optickle and degrees in Finesse.
 * Finesse generates RF sidebands when RF modulators are added to a model with the command
 ```python
 fin.addModulator(kat, 'Mod', fmod, gmod, 2, 'pm')
 ```
 This creates phase modulation (the `pm` argument) at frequency `fmod` and modulation depth `gmod`. The second to last argument specifies that the first and second order sidebands should be calculated. In contrast, Optickle defines which sidebands to compute when the model is defined and the RF modulators distribute the power accordingly. The following Optickle code creates the same frequency content as the above Finesse code
 ```python
 vRF = np.array([-2*fmod, -fmod, 0, fmod, 2*fmod])  # frequencies to track relative to the carrier
 opt = pyt.PyTickle(eng, 'opt', vRF=vRF)
 opt.addRFmodulator('Mod', fmod, 1j*gmod)
 ```
 This shows another difference. In Optickle you specify a complex modulation index. So `1j*gmod` is a phase modulation (the `pm` option with Finesse's `addModulator` function) and `gmod` is an amplitude modulation (the `am` option with Finesse).
  * Optickle adds real mirrors and beamsplitters with both an HR and AR side with its `addMirror` and `addBeamsplitter` functions, while Finesse adds fake mirrors with only a single side. Real double-sided mirror and beamsplitters are modeled in Finesse by appropriately connecting two single-sided mirrors. PyTickle's Finesse `addMirror` and `addBeamsplitter` functions add the single-sided Finesse mirrors by default but will add a compound double-sided mirror by using the `comp=True` keyword when calling the functions.
  * Optickle uses the phase convention that fields get a 180 deg phase shift on reflection from the HR surface of an optic while Finesse uses the convention that fields get a 90 deg phase shift on transmission through an optic. This leads to sometimes significant consequences when setting microscopic tunings.

## Differences in calculations

There are significant differences in how calculations are done between Optickle and Finesse, but PyTickle goes a long way in reducing these.

* Finesse calculates arbitrary higher order modes while Optickle only does TEM00, 01, and 10.
* Optickle can compute the radiation pressure modifications to the mechanical response of optics separately from the optical response while Finesse does not. PyTickle can extract the mechanical modifications from a Finesse simulation but cannot currently get the optical response separately. This is discussed in detail with the control loops.
* Optickle can only calculate physical signals that you can measure at photodiodes, the power of the sidebands at every point in the model, and the Gaussian beam parameters at every node, while Finesse can additionally compute things like the amplitudes of quantum quadratures.
